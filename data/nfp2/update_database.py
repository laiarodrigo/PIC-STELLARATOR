import sqlite3
from pathlib import Path

# Paths for the original and new databases
original_db_path = Path('data/nfp2/nfp2.db')
new_db_path = Path('data/nfp2/nfp2_converged.db')

def create_converged_database(original_db_path, new_db_path):
    try:
        # Connect to the original database
        conn_orig = sqlite3.connect(original_db_path)
        cursor_orig = conn_orig.cursor()

        # Check if the table exists and retrieve its CREATE TABLE statement
        cursor_orig.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='stellarators';")
        create_table_sql = cursor_orig.fetchone()

        if create_table_sql is None:
            raise ValueError(f"Table 'stellarators' not found in {original_db_path}.")

        create_table_sql = create_table_sql[0]  # Get the SQL statement

        # Connect to the new database
        conn_new = sqlite3.connect(new_db_path)
        cursor_new = conn_new.cursor()

        # Drop the table if it already exists
        cursor_new.execute("DROP TABLE IF EXISTS stellarators;")

        # Execute the CREATE TABLE statement in the new database
        cursor_new.execute(create_table_sql)

        # Copy rows where convergence = 1 from the original to the new table
        cursor_orig.execute("SELECT * FROM stellarators WHERE convergence = 1;")
        rows = cursor_orig.fetchall()

        # Get the number of columns to generate the placeholders dynamically
        cursor_orig.execute("PRAGMA table_info(stellarators);")
        num_columns = len(cursor_orig.fetchall())
        placeholders = ', '.join(['?'] * num_columns)

        # Insert rows into the new table
        insert_query = f"INSERT INTO stellarators VALUES ({placeholders})"
        cursor_new.executemany(insert_query, rows)

        # Commit the changes and close the connections
        conn_new.commit()
        conn_orig.close()
        conn_new.close()

        print(f"New database created at {new_db_path} with rows where convergence = 1.")

    except sqlite3.DatabaseError as e:
        print(f"Error accessing database: {e}")
    except ValueError as e:
        print(e)

# Create the new database with filtered rows
create_converged_database(original_db_path, new_db_path)
