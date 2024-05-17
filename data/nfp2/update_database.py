import sqlite3
from pathlib import Path

# Define the path to the database file
database_file = Path('data/nfp2/nfp2.db')

# Connect to the database
conn = sqlite3.connect(database_file)

# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Query to count the number of rows in the "stellarators" table
try:
    cursor.execute("SELECT COUNT(*) FROM stellarators;")
    row_count = cursor.fetchone()[0]
    print(f"Number of rows in the 'stellarators' table: {row_count}")
except sqlite3.Error as e:
    print("Error querying the database:", e)

# Commit the changes and close the connection
conn.commit()
conn.close()
