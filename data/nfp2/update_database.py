import sqlite3
from pathlib import Path

combined_db_path = Path('data/nfp2/nfp2.db')

def count_negative_quasiisodynamic(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count rows with negative quasiisodynamic
        cursor.execute("SELECT COUNT(*) FROM stellarators WHERE quasiisodynamic < 0;")
        count = cursor.fetchone()[0]
        print(f"Number of rows with negative quasiisodynamic: {count}")

        conn.close()

    except sqlite3.DatabaseError as e:
        print(f"Error accessing {db_path}: {e}")

# Count rows with negative quasiisodynamic in the database
count_negative_quasiisodynamic(combined_db_path)

