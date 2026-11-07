'''
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Purpose: To execute SQL queries and display business insights.
// - Reads SQL queries from sql/queries.sql.
// - Connects to the SQLite database.
// - Executes each query.
// - Displays query results and selected calculations, such as total revenue and customer spending rankings.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

from pathlib import Path
import sqlite3
import pandas as pd

database_path = Path("data/instamart.db")
sql_path = Path("sql/queries.sql")

sql_text = sql_path.read_text(encoding="utf-8")
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

with sqlite3.connect(database_path) as connection:
    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)

        print("\n" + "=" * 60)
        print(f"Q{number} — {len(result)} rows")
        print("=" * 60)

        if number == 2:
            print(result.to_string(index=False))
            print(
                "\nTotal revenue:",
                f"INR {result['revenue'].sum():,.2f}"
            )
        elif number == 6:
            print(
                result.sort_values(
                    ["city", "spend_rank", "customer_id"]
                ).groupby("city").head(1).to_string(index=False)
            )
            ties = (result["spend_rank"] != result["row_num"]).sum()
            print("Rows where RANK differs from ROW_NUMBER:", ties)
        elif number == 7:
            print(result.head(5).to_string(index=False))
            if not result.empty:
                print(
                    "Final cumulative revenue:",
                    f"INR {result['cumulative_revenue'].iloc[-1]:,.2f}"
                )
        else:
            print(result.head(10).to_string(index=False))

print("\nAll seven SQL queries executed.")
