"""Instamart Support Copilot: customer chat + human-review desk for support agents.

Customer page : http://127.0.0.1:8000/        (policy chat, auto hand-off to a human)
Agent desk    : http://127.0.0.1:8000/agent   (review queue, replies, resolve)

Uses only the Python standard library plus the project's own modules
(policy_rag, policy_assistant, ticket_triage.redact_pii, mock_llm).
"""

import csv
import hmac
import json
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
UI_DIR = SRC_DIR / "ui"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

import mock_llm  # noqa: E402
from policy_rag import load_policy_chunks, build_index  # noqa: E402
from policy_assistant import answer_question  # noqa: E402
from ticket_triage import redact_pii  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SEGMENTS_PATH = PROJECT_ROOT / "outputs" / "customer_segments.csv"
CASES_PATH = PROJECT_ROOT / "state" / "review_cases.json"
CHAT_LOG_PATH = PROJECT_ROOT / "outputs" / "support_chat_log.jsonl"

MAX_BODY = 10_000
MAX_MESSAGE = 1000

# ---------------------------------------------------------------------------
# Search index (built once at start-up)
# ---------------------------------------------------------------------------
CHUNKS = load_policy_chunks()
VECTORIZER, MATRIX = build_index(CHUNKS)
LLM_LOCK = threading.Lock()          # mock_llm keeps global counters


# ---------------------------------------------------------------------------
# Risk / violation rules.  Each rule -> (code, label, priority)
# ---------------------------------------------------------------------------
RULES = [
    ("FRAUD", "Possible fraud or unauthorized transaction", "high",
     r"\b(fraud\w*|unauthori[sz]ed|hacked|stolen|scam\w*|phishing|compromised"
     r"|not my transaction|(?:did not|didn't|never) (?:authori[sz]e|approve|make|do|place))\b"),
    ("ABUSE", "Abusive language or threat", "high",
     r"\b(idiot|stupid|useless|scum|bastard|shut up|f+u+c+k\w*|shit|moron|"
     r"kill|bomb|burn (?:you|down)|beat you|hurt you|i will (?:find|destroy|ruin) )"),
    ("LEGAL", "Legal or regulator threat", "high",
     r"\b(lawyer|legal action|sue|suing|court|consumer forum|consumer court|"
     r"police|defamation|rbi complaint)\b"),
    ("SECURITY", "Attempt to bypass rules or access other people's data", "high",
     r"(ignore (?:all |your |the )?(?:previous|above|prior) (?:instructions|rules)"
     r"|reveal (?:your )?(?:system )?prompt|jailbreak|bypass (?:the )?(?:policy|rules|approval|limit)"
     r"|override (?:the )?(?:policy|rules|approval|limit)"
     r"|(?:another|other|someone else'?s?) (?:customer'?s?|person'?s?|user'?s?) (?:phone|email|address|card|data|details|account)"
     r"|someone else'?s (?:phone|email|address|card|data|details|account))"),
]
HUMAN_REQUEST = re.compile(
    r"\b(human|real person|representative|supervisor|call ?back|call me|"
    r"talk to (?:a |an |some)?(?:person|agent|someone)|speak to (?:a |an |some)?(?:person|agent|someone)|"
    r"live agent|customer care executive)\b", re.I)
REFUND_INTENT = re.compile(r"\b(refund\w*|money back|reimburse\w*|compensat\w*)\b", re.I)
AMOUNT = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:rupees|rs\b|inr\b)", re.I)
ORDER_RE = re.compile(r"ORD-\d{6}", re.I)
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
CUSTOMER_ID_RE = re.compile(r"^C\d{4}$")
ORDER_ID_RE = re.compile(r"^ORD-\d{6}$")
COMPILED_RULES = [(c, l, p, re.compile(rx, re.I)) for c, l, p, rx in RULES]

NO_ANSWER_PREFIX = "I could not find"


QUESTION_START = re.compile(r"^\s*(how|what|when|who|which|why|is|are|can|do|does|will)\b", re.I)
FIRST_PERSON = re.compile(r"\b(my|i|me|we|our|someone|somebody|they|was|were|been|got|just)\b", re.I)


def is_policy_question(text):
    """True for general questions such as 'How fast must a fraud ticket reach security?'
    (a question word first and nobody reporting something that happened to them)."""
    return bool(QUESTION_START.match(text)) and not FIRST_PERSON.search(text)


def detect_flags(raw_text, masked_text):
    """Return list of flag dicts found in the customer's message."""
    flags = []
    informational = is_policy_question(raw_text)
    for code, label, priority, rx in COMPILED_RULES:
        if code == "FRAUD" and informational:
            continue
        if rx.search(raw_text):
            flags.append({"code": code, "label": label, "priority": priority})

    if REFUND_INTENT.search(raw_text):
        for m in AMOUNT.finditer(raw_text):
            value = (m.group(1) or m.group(2)).replace(",", "")
            try:
                if float(value) > 1000:
                    flags.append({
                        "code": "HIGH_VALUE_REFUND",
                        "label": "Refund above INR 1000 needs human support lead approval",
                        "priority": "normal"})
                    break
            except ValueError:
                pass

    if HUMAN_REQUEST.search(raw_text):
        flags.append({"code": "CUSTOMER_REQUEST",
                      "label": "Customer asked for a human", "priority": "normal"})
    return flags


def mask_text(text):
    """Mask email/phone/order ID (project redact_pii) and card-like numbers."""
    masked, counts = redact_pii(text)
    masked, cards = CARD_RE.subn("[CARD]", masked)
    return masked, cards


# ---------------------------------------------------------------------------
# Data look-ups (read-only)
# ---------------------------------------------------------------------------
def lookup_segment(customer_id):
    if not customer_id or not SEGMENTS_PATH.exists():
        return None
    with open(SEGMENTS_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["customer_id"] == customer_id:
                return {"segment": row["segment_name"],
                        "premium": row["is_premium"].strip().lower() == "true"}
    return None


def lookup_order(order_id):
    if not order_id or not DB_PATH.exists():
        return None
    try:
        uri = f"file:{DB_PATH.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            row = con.execute(
                """SELECT o.order_id, o.customer_id, o.status, o.order_amount,
                          o.delivery_minutes, o.quantity, p.product_name, p.category
                   FROM orders o LEFT JOIN products p ON o.product_id = p.product_id
                   WHERE o.order_id = ?""", (order_id,)).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    keys = ["order_id", "customer_id", "status", "amount", "delivery_minutes",
            "quantity", "product", "category"]
    facts = dict(zip(keys, row))
    status = str(facts["status"]).lower()
    minutes = facts["delivery_minutes"]
    amount = facts["amount"] or 0
    if status == "cancelled":
        tip = "Order was cancelled and never charged, so no refund applies (Refund Policy)."
    elif status == "completed" and minutes is not None and minutes > 45:
        tip = (f"Delivered in {minutes:.0f} min (over 45): full refund of INR {amount:,.2f} is due. "
               + ("Above INR 1000: a human support lead must approve."
                  if amount > 1000 else "Up to INR 1000: can be approved automatically.")
               + " (7-day request window not checked.)")
    elif status == "completed" and minutes is not None:
        tip = f"Delivered in {minutes:.0f} min (45 or less): no late-delivery refund."
    else:
        tip = f"Order status is '{status}'. Check manually."
    facts["suggestion"] = tip
    return facts


def refund_reply(order):
    """Customer-facing refund answer from the order record and the refund policy.
    Returns (text, flag_or_None). Only used when the customer ID matches the order."""
    oid, status = order["order_id"], str(order["status"]).lower()
    minutes, amount = order["delivery_minutes"], order["amount"] or 0
    if status == "cancelled":
        return f"Order {oid} was cancelled and never charged, so no refund applies.", None
    if status == "completed" and minutes is not None:
        if minutes <= 45:
            return (f"Order {oid} was delivered in {minutes:.0f} minutes (45 or less), "
                    "so no late-delivery refund applies."), None
        text = (f"Order {oid} was delivered in {minutes:.0f} minutes (over 45), "
                f"so it qualifies for a full refund of INR {amount:,.2f}.")
        if amount > 1000:
            return text, {"code": "HIGH_VALUE_REFUND", "priority": "normal",
                          "label": "Refund above INR 1000 needs human support lead approval"}
        return text, {"code": "REFUND_ELIGIBLE", "priority": "normal",
                      "label": "Eligible late-delivery refund to process"}
    return None, None


# ---------------------------------------------------------------------------
# Case store (JSON file, thread-safe)
# ---------------------------------------------------------------------------
STORE_LOCK = threading.RLock()


def _load_store():
    if CASES_PATH.exists():
        try:
            return json.loads(CASES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"next_id": 1, "cases": []}


def _save_store(store):
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CASES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CASES_PATH)


def sla_for(flags, premium):
    """Policy: fraud -> security team within 2 h; premium -> 1 h; everyone else 24 h."""
    fraud = any(f["code"] == "FRAUD" for f in flags)
    if premium and fraud:
        return 1, "within 1 hour (premium customer; fraud also goes to security within 2 hours)"
    if premium:
        return 1, "within 1 hour (premium customer)"
    if fraud:
        return 2, "within 2 hours (security team)"
    return 24, "within 24 hours"


def create_case(session_id, history, flags, customer_id, order_id):
    seg = lookup_segment(customer_id)
    premium = bool(seg and seg["premium"])
    order = lookup_order(order_id)
    hours, sla_text = sla_for(flags, premium)
    priority = "high" if any(f["priority"] == "high" for f in flags) else "normal"
    now = time.time()
    with STORE_LOCK:
        store = _load_store()
        case_id = f"HR-{store['next_id']:04d}"
        store["next_id"] += 1
        case = {
            "id": case_id, "session": session_id, "created": now,
            "status": "open", "priority": priority, "flags": flags,
            "reason": flags[0]["label"] if flags else "Needs human review",
            "customer_id": customer_id or None, "order_id": order_id or None,
            "segment": seg["segment"] if seg else None, "premium": premium,
            "sla_text": sla_text, "due": now + hours * 3600,
            "order": order, "assigned_to": None, "resolved": None,
            "note": None, "transcript": list(history),
        }
        store["cases"].append(case)
        _save_store(store)
    return case


def case_summary(c):
    return {k: c[k] for k in ("id", "created", "status", "priority", "reason", "customer_id",
                              "order_id", "premium", "sla_text", "due", "assigned_to")} | {
        "flags": [f["code"] for f in c["flags"]],
        "messages": len(c["transcript"]),
        "last": c["transcript"][-1]["text"][:90] if c["transcript"] else ""}


def append_to_case(case_id, role, text, extra=None):
    with STORE_LOCK:
        store = _load_store()
        for c in store["cases"]:
            if c["id"] == case_id:
                entry = {"role": role, "text": text, "ts": time.time()}
                if extra:
                    entry.update(extra)
                c["transcript"].append(entry)
                if role == "customer" and c["status"] == "resolved":
                    c["status"] = "open"        # customer wrote again: reopen
                    c["resolved"] = None
                _save_store(store)
                return c
    return None


def get_case(case_id):
    with STORE_LOCK:
        for c in _load_store()["cases"]:
            if c["id"] == case_id:
                return c
    return None


# ---------------------------------------------------------------------------
# Customer chat sessions (in memory)
# ---------------------------------------------------------------------------
SESSIONS = {}
SESSION_LOCK = threading.Lock()


def get_session(sid):
    with SESSION_LOCK:
        if sid not in SESSIONS:
            if len(SESSIONS) > 500:
                SESSIONS.pop(next(iter(SESSIONS)))
            SESSIONS[sid] = {"history": [], "fallbacks": 0, "case_id": None}
        return SESSIONS[sid]


def log_chat(session_id, role, text):
    try:
        CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "session": session_id[:8],
                                 "role": role, "text": text}, ensure_ascii=False) + "\n")
    except OSError:
        pass


HANDOFF_TEXT = {
    "FRAUD": ("I'm sorry this happened. I have marked this as high priority and sent it to our "
              "security team. Please do not share your OTP, PIN or card details with anyone."),
    "ABUSE": "I have passed this conversation to a human reviewer.",
    "LEGAL": "I have passed your message to a human reviewer who will contact you.",
    "SECURITY": ("I can't help with that request. I have passed this conversation to a human "
                 "reviewer."),
    "MISMATCH": ("I couldn't match that order to your customer ID, so a human reviewer will "
                 "check it for you."),
    "HIGH_VALUE_REFUND": "A refund above INR 1000 needs approval from a human support lead.",
    "REFUND_ELIGIBLE": ("I've sent this to our support team to process. Refunds are credited to "
                        "the original payment method within 5 working days."),
    "CUSTOMER_REQUEST": "Sure, I'm connecting you with a human agent.",
    "UNRESOLVED": ("I couldn't resolve this after a couple of tries, so I'm handing it to a human "
                   "agent with a short summary."),
}
NO_LLM_CODES = {"FRAUD", "ABUSE", "LEGAL", "SECURITY", "MISMATCH", "CUSTOMER_REQUEST"}


def handoff_payload(case):
    return {"case_id": case["id"], "priority": case["priority"],
            "sla": case["sla_text"], "reasons": [f["label"] for f in case["flags"]]}


def handle_chat(body):
    sid = str(body.get("session", ""))[:64]
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid):
        raise ValueError("Invalid session.")
    raw = str(body.get("message", "")).strip()
    if not raw:
        raise ValueError("Please type a question.")
    if len(raw) > MAX_MESSAGE:
        raise ValueError(f"Please keep your message under {MAX_MESSAGE} characters.")

    customer_id = str(body.get("customer_id", "")).strip().upper()
    order_id = str(body.get("order_id", "")).strip().upper()
    if customer_id and not CUSTOMER_ID_RE.match(customer_id):
        raise ValueError("Customer ID looks like C0001.")
    if order_id and not ORDER_ID_RE.match(order_id):
        raise ValueError("Order ID looks like ORD-123456.")
    if not order_id:
        found = ORDER_RE.search(raw)
        order_id = found.group(0).upper() if found else ""

    masked, cards = mask_text(raw)
    session = get_session(sid)
    note = None
    if cards:
        note = ("For your safety I hid the card number you typed. "
                "Never share card numbers, PINs or OTPs in chat.")

    now = time.time()
    session["history"].append({"role": "customer", "text": masked, "ts": now})
    log_chat(sid, "customer", masked)

    # Already with a human: just pass the message on.
    if session["case_id"]:
        append_to_case(session["case_id"], "customer", masked)
        case = get_case(session["case_id"])
        return {"mode": "human", "reply": None, "note": note,
                "handoff": handoff_payload(case) if case else None}

    flags = detect_flags(raw, masked)
    order_facts = lookup_order(order_id)
    if order_facts and customer_id and order_facts["customer_id"] != customer_id:
        flags.append({"code": "MISMATCH",
                      "label": "Order does not belong to the customer ID given",
                      "priority": "high"})
    order_reply = None
    if order_facts and REFUND_INTENT.search(raw):
        if customer_id and order_facts["customer_id"] == customer_id:
            order_reply, order_flag = refund_reply(order_facts)
            if order_flag and order_flag["code"] not in {f["code"] for f in flags}:
                flags.append(order_flag)
        elif not customer_id:
            note = ((note + " ") if note else "") + "Add your Customer ID so I can check that order."

    codes = {f["code"] for f in flags}
    reply, sources, confidence = None, [], None
    answer_is_fallback = False

    if order_reply and not (codes & NO_LLM_CODES):
        reply, sources = order_reply, ["refund_policy.txt"]
    elif not (codes & NO_LLM_CODES):
        with LLM_LOCK:
            mock_llm.reset()
            answer, retrieved, _tokens = answer_question(masked, CHUNKS, VECTORIZER, MATRIX)
        sources = sorted({r["source"] for r in retrieved})
        confidence = round(float(retrieved[0]["score"]), 2) if retrieved else 0.0
        if answer.startswith(NO_ANSWER_PREFIX):
            session["fallbacks"] += 1
            answer_is_fallback = True
        else:
            session["fallbacks"] = 0
        reply = answer

    if session["fallbacks"] >= 2:
        flags.append({"code": "UNRESOLVED",
                      "label": "Assistant could not resolve the question after 2 tries",
                      "priority": "normal"})

    if flags:
        text_parts = []
        primary = flags[0]["code"]
        for f in flags:
            msg = HANDOFF_TEXT.get(f["code"])
            if msg and msg not in text_parts:
                text_parts.append(msg)
        handoff_msg = text_parts[0] if text_parts else HANDOFF_TEXT["CUSTOMER_REQUEST"]
        case = create_case(sid, session["history"], flags, customer_id, order_id)
        session["case_id"] = case["id"]
        final = ((reply + "\n\n") if reply and not answer_is_fallback else "") + handoff_msg
        full = {"role": "assistant", "text": final, "ts": time.time(), "sources": sources}
        session["history"].append(full)
        append_to_case(case["id"], "assistant", final, {"sources": sources})
        log_chat(sid, "assistant", final)
        return {"mode": "human", "reply": final, "sources": sources, "confidence": confidence,
                "note": note, "handoff": handoff_payload(case)}

    session["history"].append({"role": "assistant", "text": reply, "ts": time.time(),
                               "sources": sources})
    log_chat(sid, "assistant", reply)
    return {"mode": "bot", "reply": reply, "sources": sources, "confidence": confidence,
            "note": note, "handoff": None,
            "suggest_human": answer_is_fallback}


def handle_human_request(body):
    """Customer pressed 'Talk to a human'."""
    body = dict(body)
    body["message"] = str(body.get("message") or "I would like to talk to a human agent.")
    sid = str(body.get("session", ""))
    session = get_session(sid) if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid) else None
    if session and session["case_id"]:
        case = get_case(session["case_id"])
        return {"mode": "human", "reply": None, "handoff": handoff_payload(case)}
    return handle_chat(body)


def session_history(sid):
    """What the customer chat page needs to restore itself after a refresh."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid or ""):
        return None
    with SESSION_LOCK:
        session = SESSIONS.get(sid)
    if not session:
        return {"messages": [], "case": None}
    case = get_case(session["case_id"]) if session["case_id"] else None
    source = case["transcript"] if case else session["history"]
    messages = [{"role": m["role"], "text": m["text"], "sources": m.get("sources", []),
                 "agent": m.get("agent")} for m in source]
    info = None
    if case:
        info = dict(handoff_payload(case), status=case["status"], total=len(case["transcript"]))
    return {"messages": messages, "case": info}


def customer_case_view(case_id, sid, after):
    case = get_case(case_id)
    if not case or case["session"] != sid:
        return None
    msgs = [m for m in case["transcript"][after:] if m["role"] in ("agent", "system")]
    return {"id": case["id"], "status": case["status"], "total": len(case["transcript"]),
            "messages": msgs, "assigned_to": case["assigned_to"], "sla": case["sla_text"]}


# ---------------------------------------------------------------------------
# Agent operations
# ---------------------------------------------------------------------------
PRIORITY_ORDER = {"high": 0, "normal": 1}
STATUS_ORDER = {"open": 0, "in_progress": 1, "resolved": 2}


def list_cases(status, query):
    with STORE_LOCK:
        cases = list(_load_store()["cases"])
    out = []
    q = (query or "").lower().strip()
    for c in cases:
        if status and status != "all" and c["status"] != status:
            continue
        if q:
            hay = " ".join([c["id"], c["reason"], c["customer_id"] or "", c["order_id"] or "",
                            " ".join(m["text"] for m in c["transcript"])]).lower()
            if q not in hay:
                continue
        out.append(c)
    out.sort(key=lambda c: (STATUS_ORDER[c["status"]], PRIORITY_ORDER[c["priority"]], c["due"]))
    return [case_summary(c) for c in out]


def stats():
    with STORE_LOCK:
        cases = list(_load_store()["cases"])
    now = time.time()
    active = [c for c in cases if c["status"] != "resolved"]
    return {
        "open": sum(c["status"] == "open" for c in cases),
        "in_progress": sum(c["status"] == "in_progress" for c in cases),
        "resolved": sum(c["status"] == "resolved" for c in cases),
        "high": sum(c["priority"] == "high" for c in active),
        "overdue": sum(c["due"] < now for c in active),
    }


def agent_reply(case_id, text, agent):
    text = text.strip()
    if not text or len(text) > 2000:
        raise ValueError("Reply must be 1 to 2000 characters.")
    agent = (agent or "Support agent").strip()[:40]
    with STORE_LOCK:
        store = _load_store()
        for c in store["cases"]:
            if c["id"] == case_id:
                c["transcript"].append({"role": "agent", "text": text, "ts": time.time(),
                                        "agent": agent})
                if c["status"] == "open":
                    c["status"] = "in_progress"
                c["assigned_to"] = c["assigned_to"] or agent
                _save_store(store)
                return c
    return None


def agent_set_status(case_id, status, note, agent):
    if status not in STATUS_ORDER:
        raise ValueError("Invalid status.")
    agent = (agent or "Support agent").strip()[:40]
    with STORE_LOCK:
        store = _load_store()
        for c in store["cases"]:
            if c["id"] == case_id:
                c["status"] = status
                c["assigned_to"] = c["assigned_to"] or agent
                if status == "resolved":
                    c["resolved"] = time.time()
                    c["note"] = (note or "").strip()[:500] or None
                    c["transcript"].append({"role": "system", "ts": time.time(),
                                            "text": "This case was marked resolved."})
                else:
                    c["resolved"] = None
                _save_store(store)
                return c
    return None


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
AGENT_CODE = os.environ.get("COPILOT_AGENT_CODE") or secrets.token_urlsafe(6)
CASE_PATH_RE = re.compile(r"^/api/agent/case/(HR-\d{4,})(?:/(reply|status))?$")
STATIC = {
    "/": ("customer.html", "text/html; charset=utf-8"),
    "/agent": ("agent.html", "text/html; charset=utf-8"),
    "/ui/app.css": ("app.css", "text/css; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "InstamartCopilot/2.0"

    # -- helpers
    def send_body(self, status, payload, ctype):
        data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, data):
        self.send_body(status, json.dumps(data, ensure_ascii=False),
                       "application/json; charset=utf-8")

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            raise ValueError("Invalid request size.")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid request.")
        return data

    def is_agent(self):
        given = self.headers.get("X-Agent-Code", "")
        return hmac.compare_digest(given.encode(), AGENT_CODE.encode())

    # -- routes
    def do_GET(self):
        url = urlparse(self.path)
        path, qs = url.path, parse_qs(url.query)
        try:
            if path in STATIC:
                name, ctype = STATIC[path]
                return self.send_body(200, (UI_DIR / name).read_bytes(), ctype)

            if path == "/api/history":
                data = session_history(qs.get("session", [""])[0])
                return self.send_json(200, data) if data else self.send_json(400, {"error": "Invalid session."})

            m = re.fullmatch(r"/api/case/(HR-\d{4,})", path)
            if m:
                sid = qs.get("session", [""])[0]
                after = int(qs.get("after", ["0"])[0])
                view = customer_case_view(m.group(1), sid, max(after, 0))
                return self.send_json(200, view) if view else self.send_json(404, {"error": "Not found"})

            if path.startswith("/api/agent/"):
                if not self.is_agent():
                    return self.send_json(401, {"error": "Wrong access code."})
                if path == "/api/agent/cases":
                    return self.send_json(200, {
                        "cases": list_cases(qs.get("status", ["all"])[0], qs.get("q", [""])[0]),
                        "stats": stats(), "server_time": time.time()})
                m = CASE_PATH_RE.match(path)
                if m and not m.group(2):
                    case = get_case(m.group(1))
                    return self.send_json(200, case) if case else self.send_json(404, {"error": "Not found"})
            self.send_body(404, "Page not found", "text/plain; charset=utf-8")
        except Exception as err:                      # noqa: BLE001
            print("GET failed:", repr(err))
            self.send_json(500, {"error": "Something went wrong."})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_json()
            if path == "/api/chat":
                return self.send_json(200, handle_chat(body))
            if path == "/api/human":
                return self.send_json(200, handle_human_request(body))

            m = CASE_PATH_RE.match(path)
            if m and m.group(2):
                if not self.is_agent():
                    return self.send_json(401, {"error": "Wrong access code."})
                if m.group(2) == "reply":
                    case = agent_reply(m.group(1), str(body.get("text", "")), body.get("agent"))
                else:
                    case = agent_set_status(m.group(1), str(body.get("status", "")),
                                            str(body.get("note", "")), body.get("agent"))
                return self.send_json(200, case) if case else self.send_json(404, {"error": "Not found"})
            self.send_json(404, {"error": "Endpoint not found"})
        except ValueError as err:
            self.send_json(400, {"error": str(err)})
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Invalid request."})
        except Exception as err:                      # noqa: BLE001
            print("POST failed:", repr(err))
            self.send_json(500, {"error": "Something went wrong. Please try again."})

    def log_message(self, fmt, *args):
        print("Web request:", fmt % args)


def serve(host="127.0.0.1", port=8000, on_ready=None):
    server = ThreadingHTTPServer((host, port), Handler)
    print("=" * 60)
    print("Instamart Support Copilot is ready")
    print(f"  Customer chat : http://{host}:{port}/")
    print(f"  Agent desk    : http://{host}:{port}/agent")
    print(f"  Agent access code: {AGENT_CODE}")
    print("  (set your own with the COPILOT_AGENT_CODE environment variable)")
    print("  Stop the server with Ctrl+C.")
    print("=" * 60)
    if on_ready:
        on_ready()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Instamart Support Copilot...")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
