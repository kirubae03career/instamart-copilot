"""Run the Instamart Customer Intelligence & Support Copilot.

Usage (from the project root, with the virtual environment active):

    python run_project.py                 # start the web app (customer chat + agent desk)
    python run_project.py --check         # only check that everything is in place
    python run_project.py --pipeline      # run Parts A-C + policy checks first, then start the app
    python run_project.py --pipeline-only # run the pipeline and stop
    python run_project.py --with-refund-agent   # also run the refund agent scenarios
    python run_project.py --port 8080 --no-browser

The pipeline never regenerates the raw data (generate_data.py is not run), so your
datasets stay as they are.
"""

import argparse
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PIPELINE = [
    ("Part A", "src/check_raw_data.py"),
    ("Part A", "src/clean_data.py"),
    ("Part A", "src/validate_data.py"),
    ("Part A", "src/check_database.py"),
    ("Part A", "src/run_sql_queries.py"),
    ("Part A", "src/eda.py"),
    ("Part B", "src/rfm_analysis.py"),
    ("Part B", "src/customer_segmentation.py"),
    ("Part B", "src/lapse_model.py"),
    ("Part C", "src/ticket_triage.py"),
    ("Part C", "src/build_triage_outputs.py"),
    ("Part C", "src/evaluate_triage.py"),
    ("Part D", "src/policy_rag.py"),
    ("Part D", "src/policy_assistant.py"),
    ("Part D", "src/compare_policy_overlap.py"),
]
REFUND_AGENT = ("Part D", "src/refund_agent.py")

REQUIRED_FILES = [
    "mock_llm.py", "src/policy_rag.py", "src/policy_assistant.py", "src/ticket_triage.py",
    "src/support_app.py", "src/ui/customer.html", "src/ui/agent.html", "src/ui/app.css",
]
RAW_FILES = ["customers.csv", "products.csv", "orders.csv", "tickets.csv"]


def check_environment():
    """Print problems that would stop the project. Return True when all is well."""
    ok = True
    print("Checking project...")

    if sys.version_info < (3, 9):
        print("  [!] Python 3.9 or newer is required (3.11+ recommended).")
        ok = False

    for name in ("pandas", "numpy", "sklearn", "joblib"):
        try:
            __import__(name)
        except ImportError:
            print(f"  [!] Missing package '{name}'. Run: python -m pip install pandas numpy scikit-learn joblib")
            ok = False

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            print(f"  [!] Missing file: {rel}")
            ok = False

    policies = list((ROOT / "data" / "policies").glob("*.txt"))
    if not policies:
        print("  [!] No policy files found in data/policies/")
        ok = False

    for name in RAW_FILES:
        path = ROOT / "data" / "raw" / name
        if not path.exists():
            print(f"  [!] Missing raw data file: data/raw/{name} (run: python generate_data.py)")
            ok = False
            continue
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        if name == "products.csv" and first and not first[0].startswith("product_id"):
            print("  [!] data/raw/products.csv line 1 must start with 'product_id'. "
                  "Delete any extra text before it (for example 'requirements.txt').")
            ok = False

    if not (ROOT / "data" / "instamart.db").exists():
        print("  [i] data/instamart.db not found. The chat works, but order details for agents "
              "need the database. Run: python run_project.py --pipeline")
    if not (ROOT / "outputs" / "customer_segments.csv").exists():
        print("  [i] outputs/customer_segments.csv not found. Premium-customer detection is off "
              "until you run the pipeline.")

    print("  All checks passed." if ok else "  Fix the items marked [!] and run again.")
    return ok


def run_pipeline(with_refund_agent=False):
    steps = list(PIPELINE) + ([REFUND_AGENT] if with_refund_agent else [])
    print(f"\nRunning the pipeline ({len(steps)} steps). Output is shown for each step.\n")
    start = time.time()
    for number, (part, script) in enumerate(steps, start=1):
        print("=" * 70)
        print(f"[{number}/{len(steps)}] {part}: {script}")
        print("=" * 70, flush=True)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[!] Step failed: {script} (exit code {result.returncode}). Stopping.")
            return False
    print(f"\nPipeline finished in {time.time() - start:.0f} seconds.\n")
    return True


def start_app(port, open_browser):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import support_app

    def open_pages():
        if open_browser:
            threading.Timer(0.8, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()

    support_app.serve(host="127.0.0.1", port=port, on_ready=open_pages)


def main():
    parser = argparse.ArgumentParser(description="Run the Instamart Support Copilot project.")
    parser.add_argument("--check", action="store_true", help="only check the setup")
    parser.add_argument("--pipeline", action="store_true", help="run the analysis pipeline, then start the app")
    parser.add_argument("--pipeline-only", action="store_true", help="run the analysis pipeline and stop")
    parser.add_argument("--with-refund-agent", action="store_true",
                        help="include the refund agent scenarios in the pipeline")
    parser.add_argument("--port", type=int, default=8000, help="web port (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)
    if args.check:
        return

    if args.pipeline or args.pipeline_only:
        if not run_pipeline(args.with_refund_agent):
            sys.exit(1)
        if args.pipeline_only:
            return

    start_app(args.port, not args.no_browser)


if __name__ == "__main__":
    main()
