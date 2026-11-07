'''
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Purpose: To clean the data and store it for analysis.
// - Standardizes customer city and gender values.
// - Handles invalid or missing customer ages.
// - Converts dates and amounts into appropriate data types.
// - Removes duplicate orders and keeps the latest record for each order ID.
// - Fills missing order amounts and eligible delivery times.
// - Saves cleaned CSV files and creates SQLite database tables.
// - Prints a summary of the cleaned data.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

from pathlib import Path
import sqlite3

import pandas as pd

# Project folders
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. CLEAN CUSTOMERS
# --------------------------------------------------
customers = pd.read_csv(RAW_DIR / "customers.csv")

# Standardize city names
customers["city"] = customers["city"].astype("string").str.strip().str.title()

# Standardize gender values
gender_map = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
}

gender_normalized = (
    customers["gender"].astype("string").str.strip().str.lower()
)

customers["gender"] = gender_normalized.map(gender_map).fillna("unknown")

# Convert age to numeric and treat ages outside 10-100 as missing
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# Fill missing ages using the median age within each city
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# Convert signup date
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# --------------------------------------------------
# 2. LOAD PRODUCTS
# --------------------------------------------------
products = pd.read_csv(RAW_DIR / "products.csv")

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="raise"
)

# --------------------------------------------------
# 3. CLEAN ORDERS
# --------------------------------------------------
orders = pd.read_csv(RAW_DIR / "orders.csv")

# i. Convert order timestamp
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# ii. Remove exact duplicate rows
before_duplicates = len(orders)
orders = orders.drop_duplicates().copy()
duplicates_removed = before_duplicates - len(orders)

# iii. Keep only the latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset=["order_id"], keep="last"
).copy()

# iv. Convert amounts to numbers
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Get the catalogue price and category for each product
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

# Fill missing amounts using price, quantity and discount
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# v. Treat negative delivery times as missing
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0, "delivery_minutes"
] = float("nan")

# Use completed orders to calculate category median delivery times
completed_mask = orders["status"].str.lower().eq("completed")

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

# Align category medians to the original order index
completed_orders = orders.loc[completed_mask].copy()
completed_orders["category_median"] = (
    completed_orders.groupby("category")["delivery_minutes"]
    .transform("median")
)

# Fill missing delivery times for completed orders only
orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"].fillna(
        completed_orders["category_median"]
    )
)

# Cancelled orders retain missing delivery times.
# Remove the temporary product lookup columns from orders.
orders = orders.drop(columns=["price_inr", "category"])

# --------------------------------------------------
# 4. SAVE CLEAN CSV FILES
# --------------------------------------------------
customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)

# Store timestamps in the requested text format
orders_to_save = orders.copy()
orders_to_save["order_ts"] = orders_to_save["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
orders_to_save.to_csv(CLEAN_DIR / "orders.csv", index=False)

# --------------------------------------------------
# 5. SAVE TABLES TO SQLITE
# --------------------------------------------------
database_path = Path("data/instamart.db")

with sqlite3.connect(database_path) as connection:
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders_to_save.to_sql("orders", connection, if_exists="replace", index=False)

# --------------------------------------------------
# 6. PRINT SUMMARY
# --------------------------------------------------
print("Customer rows:", len(customers))
print("Product rows:", len(products))
print("Order rows:", len(orders))
print("Exact duplicate rows removed:", duplicates_removed)
print("Order status counts:")
print(orders["status"].value_counts().to_string())

print("Missing customer ages:", customers["age"].isna().sum())
print("Missing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery times among completed orders:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum(),
)
print(
    "Missing delivery times among cancelled orders:",
    orders.loc[
        orders["status"].str.lower().eq("cancelled"),
        "delivery_minutes",
    ].isna().sum(),
)

completed_revenue = orders.loc[
    completed_mask, "order_amount"
].sum()

print(f"Completed revenue: INR {completed_revenue:,.2f}")
print("Clean CSV files and SQLite database saved.")