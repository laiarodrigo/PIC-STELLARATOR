import sqlite3
from pathlib import Path

combined_db_path = Path('data/nfp2/nfp2.db')

def print_first_eight_columns(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Select the first eight columns from the first 10 rows where convergence = 1
        cursor.execute("""
            SELECT * FROM stellarators WHERE convergence = 1 LIMIT 10;
        """)
        rows = cursor.fetchall()
        
        # Assuming the table has at least 8 columns
        for row in rows:
            print(row[:8])

        conn.close()

    except sqlite3.DatabaseError as e:
        print(f"Error accessing {db_path}: {e}")

# Print the first eight columns from the first 10 lines where convergence = 1
print_first_eight_columns(combined_db_path)
