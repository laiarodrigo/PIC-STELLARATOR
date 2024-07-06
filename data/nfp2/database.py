import sqlite3
from pathlib import Path


# Define the path to the database folder
database_folder = Path('data/nfp2')

# Define the path to the database file within the specific field period directory
database_file = database_folder / 'example.db'
conn = sqlite3.connect(database_file)

# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Create a table to store the stellarator data
cursor.execute("""CREATE TABLE IF NOT EXISTS examples(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rbc_0_0 REAL,
            rbc_1_0 REAL, 
            rbc_m1_1 REAL, 
            rbc_0_1 REAL,
            rbc_1_1 REAL, 
            zbs_0_0 REAL,
            zbs_1_0 REAL, 
            zbs_m1_1 REAL,
            zbs_0_1 REAL,
            zbs_1_1 REAL,
            quasisymmetry REAL, 
            quasiisodynamic REAL, 
            rotational_transform REAL, 
            inverse_aspect_ratio REAL,
            mean_local_magnetic_shear REAL, 
            vacuum_magnetic_well REAL, 
            maximum_elongation REAL, 
            mirror_ratio REAL, 
            number_of_field_periods_nfp REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")

conn.commit()
conn.close()

