
import sqlite3
from pathlib import Path

# Go up one level from src to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

print("Project root:", PROJECT_ROOT)
print("Database path:", DB_PATH)
print("Database exists:", DB_PATH.exists())

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

tables = cursor.execute("""
    SELECT name, sql
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""").fetchall()

print("\nTables and schemas:")

for table_name, schema in tables:
    print(f"\nTable: {table_name}")
    print(schema)

connection.close()
