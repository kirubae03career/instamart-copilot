'''
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Purpose: To inspect raw data before cleaning.
// - Reads customers.csv, products.csv, orders.csv, and tickets.csv.
// - Displays the number of rows and columns.
// - Shows the first three records.
// - Displays column data types and missing values.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

from pathlib import Path
import pandas as pd

data_dir = Path("data/raw")

files = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
]

for filename in files:
    file_path = data_dir / filename

    print(f"\n{'=' * 40}")
    print(filename)

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())