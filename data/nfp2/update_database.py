import sqlite3
from pathlib import Path

# Define the path to the database file
database_file = Path('../../data/nfp2/nfp2.db')

# Connect to the database
conn = sqlite3.connect(database_file)

# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Update the "convergence" column to 0 where "quasisymmetry" > 10
try:
    cursor.execute("UPDATE stellarators SET convergence = 0 WHERE quasisymmetry > 10;")
    print("Convergence values updated successfully.")
except sqlite3.Error as e:
    print("Error updating convergence values:", e)

# Query to count the number of rows where "convergence" is 1
try:
    cursor.execute("SELECT COUNT(*) FROM stellarators WHERE convergence = 1;")
    convergence_count = cursor.fetchone()[0]
    print(f"Number of rows where 'convergence' is 1: {convergence_count}")
except sqlite3.Error as e:
    print("Error querying the database:", e)

# Commit the changes and close the connection
conn.commit()
conn.close()
