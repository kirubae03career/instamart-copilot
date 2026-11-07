'''
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Purpose: To perform Exploratory Data Analysis (EDA) on cleaned Instamart data and discover useful business insights.
// EDA means examining data to understand patterns, revenue, relationships, and unusual values before building machine-learning models.
// What each section does
// Section	Purpose
// 1. Load data	           : Reads cleaned orders, customers, and products CSV files using pandas.
// 2. Data shapes	       : Displays the number of rows and columns in each dataset.
// 3. Revenue by city	   : Calculates total revenue, average order amount, and order count for each city, considering completed orders only.
// 4. Revenue by category  : Calculates total revenue and order count for each product category.
// 5. Order statistics	   : Calculates mean, median, minimum, maximum, standard deviation, and skewness of completed order amounts.
// 6. Outlier detection	   : Uses the IQR method to identify unusually low or high order amounts.
// 7. Correlation analysis : Measures relationships between order amount, quantity, and discount percentage.
// 8. Pivot table	       : Summarizes revenue by city and product category, including grand totals.
// 9. Save results         : Exports city revenue, category revenue, and the pivot table as CSV files in the outputs folder.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

from pathlib import Path
import pandas as pd

# Project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load cleaned CSV files
orders = pd.read_csv(PROJECT_ROOT / "data" / "clean" / "orders.csv")
customers = pd.read_csv(PROJECT_ROOT / "data" / "clean" / "customers.csv")
products = pd.read_csv(PROJECT_ROOT / "data" / "clean" / "products.csv")

print("=" * 60)
print("INSTAMART CUSTOMER COPILOT - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Check the data
print("\n1. DATA SHAPES")
print("Orders:", orders.shape)
print("Customers:", customers.shape)
print("Products:", products.shape)

# 2. Revenue by city
orders_customers = orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

completed_orders = orders_customers[
    orders_customers["status"].str.lower() == "completed"
].copy()

city_revenue = (
    completed_orders.groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        order_count=("order_id", "count")
    )
    .sort_values("total_revenue", ascending=False)
)

print("\n2. REVENUE BY CITY")
print(city_revenue.round(2).to_string())

# 3. Revenue by product category
orders_products = completed_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_revenue = (
    orders_products.groupby("category")
    .agg(
        total_revenue=("order_amount", "sum"),
        order_count=("order_id", "count")
    )
    .sort_values("total_revenue", ascending=False)
)

print("\n3. REVENUE BY PRODUCT CATEGORY")
print(category_revenue.round(2).to_string())

# 4. Order amount statistics
amounts = completed_orders["order_amount"]

print("\n4. ORDER AMOUNT STATISTICS")
print(f"Mean: INR {amounts.mean():.2f}")
print(f"Median: INR {amounts.median():.2f}")
print(f"Minimum: INR {amounts.min():.2f}")
print(f"Maximum: INR {amounts.max():.2f}")
print(f"Standard deviation: INR {amounts.std():.2f}")
print(f"Skewness: {amounts.skew():.2f}")

print("\nEDA STEP 1 COMPLETED")


# 5. Outlier detection using the IQR method
print("\n5. OUTLIER DETECTION")

Q1 = amounts.quantile(0.25)
Q3 = amounts.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

low_outliers = amounts[amounts < lower_limit]
high_outliers = amounts[amounts > upper_limit]

print(f"Q1: INR {Q1:.2f}")
print(f"Q3: INR {Q3:.2f}")
print(f"IQR: INR {IQR:.2f}")
print(f"Lower limit: INR {lower_limit:.2f}")
print(f"Upper limit: INR {upper_limit:.2f}")
print(f"Low outliers: {len(low_outliers)}")
print(f"High outliers: {len(high_outliers)}")


# 6. Correlation analysis
print("\n6. CORRELATION ANALYSIS")

numeric_columns = [
    "order_amount",
    "quantity",
    "discount_pct"
]

correlation_matrix = completed_orders[numeric_columns].corr()

print(correlation_matrix.round(3).to_string())


# 7. Pivot table: revenue by city and category
print("\n7. PIVOT TABLE: CITY VS CATEGORY")

pivot = pd.pivot_table(
    orders_products,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2).to_string())

print("\nGrand total revenue:")
print(f"INR {pivot.loc['Grand Total', 'Grand Total']:.2f}")


# 8. Save EDA results
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

city_revenue.to_csv(output_dir / "revenue_by_city.csv")
category_revenue.to_csv(output_dir / "revenue_by_category.csv")
pivot.to_csv(output_dir / "revenue_pivot_city_category.csv")

print("\n8. EDA RESULTS SAVED")
print("Saved to:", output_dir)