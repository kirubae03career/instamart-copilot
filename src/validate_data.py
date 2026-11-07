'''
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Purpose: To verify the cleaned data and database.
// - Checks customer, product, and order row counts.
// - Verifies order ID uniqueness.
// - Checks customer ages, city names, and gender counts.
// - Validates order statuses, missing values, and total revenue.
// - Checks that the required SQLite tables contain the expected number of rows.
// - Prints ALL CHECKPOINTS PASSED if all assertions succeed.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

from pathlib import Path
import sqlite3
import pandas as pd

# Load cleaned CSV files
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Check row counts and order uniqueness
assert len(customers) == 1200, "Customer row count mismatch"
assert len(products) == 40, "Product row count mismatch"
assert len(orders) == 12309, "Order row count mismatch"
assert orders["order_id"].nunique() == 12309, "Duplicate order IDs found"

# Check customer cleaning
assert customers["age"].isna().sum() == 0, "Missing customer ages found"
assert customers["age"].between(18, 60).all(), "Age range mismatch"
assert set(customers["city"].unique()) == {
    "Pune", "Delhi", "Bengaluru",
    "Chennai", "Hyderabad", "Mumbai"
}, "City names mismatch"

assert customers["gender"].value_counts().to_dict() == {
    "female": 610,
    "male": 560,
    "unknown": 30,
}, "Gender counts mismatch"

# Check order statuses
status_counts = orders["status"].str.lower().value_counts().to_dict()

assert status_counts.get("completed", 0) == 11110, "Completed count mismatch"
assert status_counts.get("cancelled", 0) == 1199, "Cancelled count mismatch"
assert "pending" not in status_counts, "Pending orders remain"

# Check missing values and revenue
assert orders["order_amount"].notna().all(), "Missing order amounts found"

completed = orders["status"].str.lower().eq("completed")
cancelled = orders["status"].str.lower().eq("cancelled")

assert orders.loc[completed, "delivery_minutes"].notna().all(), (
    "Completed orders have missing delivery times"
)
assert orders.loc[cancelled, "delivery_minutes"].isna().all(), (
    "Cancelled orders should have missing delivery times"
)

revenue = orders.loc[completed, "order_amount"].sum()
assert abs(revenue - 2998668.80) < 0.01, "Revenue mismatch"

# Check SQLite tables and row counts
with sqlite3.connect("data/instamart.db") as connection:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        connection,
    )["name"].tolist()

    assert {"customers", "products", "orders"}.issubset(set(tables)), (
        "Required database tables are missing"
    )

    db_customers = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM customers", connection
    ).iloc[0]["n"]

    db_products = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM products", connection
    ).iloc[0]["n"]

    db_orders = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM orders", connection
    ).iloc[0]["n"]

    assert db_customers == 1200, "Database customer count mismatch"
    assert db_products == 40, "Database product count mismatch"
    assert db_orders == 12309, "Database order count mismatch"

print("ALL CHECKPOINTS PASSED")
