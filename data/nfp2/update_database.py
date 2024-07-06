import sqlite3
from pathlib import Path

combined_db_path = Path('data/nfp2/example.db')

def format_real_columns_to_five_decimals(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch the table information
        cursor.execute("PRAGMA table_info(examples);")
        columns = cursor.fetchall()

        # List of REAL columns
        real_columns = [column[1] for column in columns if column[2] == 'REAL']

        # Update each REAL column to have five decimal places
        for column in real_columns:
            cursor.execute(f"""
                UPDATE examples 
                SET {column} = ROUND({column}, 5);
            """)
        
        conn.commit()
        print("All REAL columns formatted to 5 decimal places.")

        conn.close()

    except sqlite3.DatabaseError as e:
        print(f"Error accessing {db_path}: {e}")

# Format REAL columns in the database
format_real_columns_to_five_decimals(combined_db_path)
