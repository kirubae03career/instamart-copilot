# Instamart Customer Intelligence & Support Copilot

An offline, end-to-end data project on synthetic quick-commerce data: clean data, segment customers, predict churn, triage support tickets, and answer policy questions with a small RAG assistant. Everything runs locally with a mock LLM, so **no API key or internet is needed**.

---

## Learning Outcomes

After working through this project you will be able to:

- Clean messy data (duplicates, bad numbers, negative values, inconsistent spellings, missing values) and explain each cleaning decision.
- Load clean data into **SQLite** and answer business questions with SQL (`JOIN`, `GROUP BY`, `HAVING`).
- Build **RFM** features (Recency, Frequency, Monetary) and group customers with **KMeans**, choosing K with **silhouette scores** and viewing clusters with **PCA**.
- Train a **churn (lapse) model** without data leakage, tune it with grid search, and handle class imbalance with oversampling.
- Classify support tickets with an LLM using **zero-shot vs few-shot** prompting, while **redacting PII** and enforcing a token budget.
- Build a **Retrieval-Augmented Generation (RAG)** pipeline: chunking, TF-IDF retrieval, grounded answers, and escalation when the answer is not in the policy.
- Build a tool-using **agent** with iteration limits, checkpoints, crash recovery and safety rules.
- Work with Jupyter notebooks in VS Code.

---

## Setup and Runs

Run these in **Windows PowerShell** from the project root.

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (ipykernel is needed to run notebooks)
python -m pip install pandas numpy scikit-learn joblib ipykernel

# 3. (Optional) regenerate the synthetic raw data
#    Warning: this overwrites data/raw/
python .\generate_data.py
```

### Option 1: Run the notebooks (recommended for learning)

Open the four notebooks in order (see [How to Open a Notebook](#how-to-open-a-notebook)) and use **Run All**.

| Order | Notebook                              |
|-------|---------------------------------------|
| 1     | `Part_A_Data_Foundation.ipynb`        |
| 2     | `Part_B_Segmentation_and_Churn.ipynb` |
| 3     | `Part_C_Ticket_Triage.ipynb`          |
| 4     | `Part_D_Policy_RAG_Copilot.ipynb`     |

Run Part A first: Parts B and C read the cleaned files in `data/clean/`.

### Option 2: Run the scripts

```powershell
python .\src\check_raw_data.py
python .\src\clean_data.py
python .\src\validate_data.py
python .\src\check_database.py
python .\src\run_sql_queries.py
python .\src\eda.py
python .\src\rfm_analysis.py
python .\src\customer_segmentation.py
python .\src\lapse_model.py
python .\src\ticket_triage.py
python .\src\build_triage_outputs.py
python .\src\evaluate_triage.py
python .\src\policy_rag.py
python .\src\policy_assistant.py
python .\src\compare_policy_overlap.py
python .\src\refund_agent.py
```

The notebooks contain the same code as these scripts, so both options give the same results.

> **Note:** `refund_agent` saves state in `state/agent_checkpoint.json` and `outputs/refund_ledger.jsonl`. Do not delete them if you want to reproduce the recorded refund results.

---

## Extensions Needed in VS Code

| Extension  | Publisher  | Why                                               |
|------------|------------|---------------------------------------------------|
| **Python** | Microsoft  | Python language support and interpreter selection |
| **Jupyter** | Microsoft | Opens and runs `.ipynb` notebooks inside VS Code  |

Install them from the Extensions panel (`Ctrl+Shift+X`). The Jupyter extension also needs the `ipykernel` package in your virtual environment (installed in the setup step above).

---

## How to Open a Notebook

1. Open the project folder in VS Code (**File > Open Folder**).
2. Click a `.ipynb` file in the Explorer panel.
3. Click **Select Kernel** (top right) > **Python Environments** > choose the project's **`.venv`**.
4. Click **Run All**, or press `Shift+Enter` to run one cell at a time.

If the `.venv` is not listed, press `Ctrl+Shift+P`, run **Python: Select Interpreter**, choose the `.venv`, then select the kernel again.

The first cell of each notebook finds the project root and sets the working folder, so the notebooks work from the project root or from a sub-folder.

---

## Repository Map

```
instamart-copilot/
├── Part_A_Data_Foundation.ipynb
├── Part_B_Segmentation_and_Churn.ipynb
├── Part_C_Ticket_Triage.ipynb
├── Part_D_Policy_RAG_Copilot.ipynb
├── generate_data.py          # creates the synthetic raw data
├── mock_llm.py               # local mock LLM (no API needed)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                  # messy input: customers, orders, products, tickets, policy_eval, agent_scenarios
│   ├── clean/                # cleaned CSVs (output of Part A)
│   ├── policies/             # 5 policy text files used by RAG
│   └── instamart.db          # SQLite database
├── sql/
│   └── queries.sql           # business queries
├── src/
│   ├── Part A: check_raw_data.py, clean_data.py, validate_data.py,
│   │           check_database.py, run_sql_queries.py, eda.py
│   ├── Part B: rfm_analysis.py, customer_segmentation.py, lapse_model.py
│   ├── Part C: ticket_triage.py, build_triage_outputs.py, evaluate_triage.py
│   ├── Part D: policy_rag.py, policy_assistant.py,
│   │           compare_policy_overlap.py, refund_agent.py
│   └── web.py                # web interface (not part of the notebooks)
├── outputs/                  # result CSVs, logs, saved model
└── state/                    # agent checkpoint
```

---

## Part A Summary: Data Foundation

**Goal:** clean messy data and load it into a database.

**What it does**
- Inspects raw files, then cleans customers, products and orders.
- Standardizes city names (`.strip().title()`) and gender values (`m/man` to male, `f/woman` to female, otherwise `unknown`).
- Treats ages outside 10 to 100 as missing, then fills them with the **median age of the customer's city**.
- Removes exact duplicate orders, then keeps the latest record per `order_id`.
- Fills missing order amounts from price x quantity x discount.
- Treats negative delivery times as missing. Missing delivery times are filled with the **category median for completed orders only**; cancelled orders stay missing.
- Saves clean CSVs and SQLite tables (`customers`, `products`, `orders`), then validates everything with assertions.

**Results**

| Check                                        | Value               |
|----------------------------------------------|---------------------|
| Customers / products / orders after cleaning | 1,200 / 40 / 12,309 |
| Exact duplicate order rows removed           | 119                 |
| Completed / cancelled orders                 | 11,110 / 1,199      |
| Completed revenue                            | INR 2,998,668.80    |

---

## Part B Summary: Segmentation and Churn

**Goal:** Group customers and predict who will stop ordering.

**What it does**
- **RFM:** 1,175 customers analyzed, using 1 October 2025 as the reference date.
- **KMeans:** tests K = 2 to 8. **K = 4** has the best silhouette score (**0.526**), giving four segments.
- **PCA:** two components explain about **85%** of the variance, used to visualize the clusters.
- **Lapse model:** features come only from orders before 1 September 2025; September activity is used only to create the label (avoids data leakage). Compares Logistic Regression and Random Forest with grid search.

**Results**

| Segment    | Customers |
|------------|-----------|
| Regular    | 436       |
| High Value | 419       |
| Budget     | 182       |
| Premium    | 138       |

| Lapse model (test set)             | Precision | Recall | F1    |
|------------------------------------|-----------|--------|-------|
| Logistic Regression                | 0.644     | 0.652  | 0.648 |
| Random Forest                      | 0.587     | 0.607  | 0.597 |
| Logistic Regression + oversampling | 0.600     | 0.809  | 0.689 |

Cohort: 1,156 customers, of which 38.3% lapsed. Oversampling raises recall (catching more customers who will lapse) at the cost of some precision.

---

## Part C Summary: Ticket Triage

**Goal:** classify support tickets and route them.

**What it does**
- **Redacts PII first** (email, phone, order ID) before any text reaches the LLM.
- Classifies each ticket as `billing`, `delivery`, `technical` or `fraud`.
- Compares **zero-shot** (no examples) with **few-shot** (examples from past tickets).
- Validates the model's output, retries with backoff, and enforces a **token budget** (a 6,000-token demo shows tickets being stopped when the budget runs out).

**Results (147 tickets)**

| Mode      | Accuracy |
|-----------|----------|
| Zero-shot | 85.71%   |
| Few-shot  | 100.00%  |

Zero-shot confuses some technical and fraud tickets with billing; examples in the prompt remove those errors.

---

## Part D Summary: Policy RAG Co-Pilot

**Goal:** answer customer questions from company policies.

**What it does**
- **Retrieve:** splits 5 policy files into chunks (40 words, 10 words overlap), builds a TF-IDF index and ranks chunks by cosine similarity.
- **Generate:** the mock LLM answers using only the retrieved chunks.
- **Escalate:** if the policy has no answer (for example, a cricket question), it replies that it could not find the information instead of guessing.
- **Compare overlap:** re-runs the evaluation with no chunk overlap.
- **Refund agent:** uses tools (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`) with iteration, token and timeout limits, a crash-safe checkpoint, and human escalation for large refunds (above INR 1000).

**Results (14 evaluation questions)**

| Setting         | Right document retrieved | Expected fact found | Correct out-of-scope fallback |
|-----------------|--------------------------|---------------------|-------------------------------|
| 10-word overlap | 12/12                    | 12/12               | 2/2                           |
| No overlap      | 12/12                    | 9/12                | 2/2                           |

Overlap keeps sentences from being cut at chunk edges. The agent scenarios cover refund, escalation, denial, not found, and fallbacks for max iterations, token limit and timeout.

---

## GitHub Suggestions

- **Add a `.gitignore`** so the environment and caches are not committed:
  ```gitignore
  .venv/
  __pycache__/
  .ipynb_checkpoints/
  *.pyc
  ```
  The `.venv` folder is large and machine-specific, so never commit it.
- **Pin versions** in `requirements.txt` (for example `pandas==3.0.6`, `scikit-learn==1.9.1`) so results are reproducible. Add `ipykernel`.
- **Decide what to track:** `data/raw/` and `data/policies/` are inputs and belong in the repo. `data/clean/`, `data/instamart.db` and `outputs/` can be regenerated, so either ignore them or keep them only as reference results.
- **Clear notebook outputs** before committing (**Clear All Outputs**) to keep diffs small, or commit them deliberately if you want results visible on GitHub.
- **Use clear commits**, for example `Part A: clean data and build SQLite DB`, and one commit per part.
- **Add a `LICENSE`** and a short description and topics on the GitHub repo page.
- **Check line 1 of `data/raw/products.csv`** before pushing: it must start with `product_id`, with no extra text in front of it.

---

## Limitations and Next Steps

**Limitations**
- The data is **synthetic**, so patterns are cleaner than real life and results will not transfer directly.
- The **mock LLM** is rule-based. The 100% few-shot accuracy and 12/12 policy score reflect a controlled setup, not a real LLM's behavior.
- The lapse model is modest (F1 about 0.65 to 0.69) on a small cohort of 1,156 customers, with a single time split and no cross-time validation.
- RFM segments use the full order history, so the retention list is an illustrative post-hoc analysis, not a forecast.
- TF-IDF matches words, not meaning, so paraphrased questions may retrieve the wrong chunk. The policy set is also tiny (5 files).
- No automated tests beyond the data assertions, and no charts are saved by default.

**Next steps**
- Swap `mock_llm.py` for a real LLM API and re-measure triage and RAG quality.
- Replace TF-IDF with **embedding-based retrieval** and a vector store; add re-ranking.
- Try gradient boosting for churn, add more features (basket size, time between orders), and use time-based cross-validation and probability calibration.
- Add plots (cluster scatter from PCA, confusion matrices, feature importance) to the notebooks.
- Add unit tests (`pytest`) for PII redaction, refund calculation and chunking.
- Wrap the pipeline with a single command (or Makefile) and add CI on GitHub to run the checks automatically.
- Serve the copilot through `src/web.py` or a small app with logging and human-review queues.