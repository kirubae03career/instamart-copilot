
from pathlib import Path
import pandas as pd

# Locate the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load cleaned orders
orders = pd.read_csv(
    PROJECT_ROOT / "data" / "clean" / "orders.csv"
)

# Convert order timestamps to dates
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Keep completed orders only
completed = orders[
    orders["status"].str.lower() == "completed"
].copy()

# Fixed snapshot date for reproducible results
snapshot_date = pd.Timestamp("2025-10-01")

# Calculate RFM values for each customer
rfm = completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
)

# Recency: days since the latest completed order
rfm["recency"] = (
    snapshot_date - rfm["last_order_date"]
).dt.days

# Keep only the RFM columns
rfm = rfm[["recency", "frequency", "monetary"]]

# Save the RFM results
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

rfm.to_csv(output_dir / "customer_rfm.csv")

# Display a summary
print("=" * 50)
print("CUSTOMER RFM ANALYSIS")
print("=" * 50)

print("\nNumber of customers:", len(rfm))
print("\nRFM sample:")
print(rfm.head(10).round(2).to_string())

print("\nRFM statistics:")
print(rfm.describe().round(2).to_string())

print("\nSaved file:")
print(output_dir / "customer_rfm.csv")
