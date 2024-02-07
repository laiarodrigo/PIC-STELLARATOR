import sqlite3
import pandas as pd

database = '/data/nfp2/nfp2.db'

# Connect to the database (create if not exists)
conn = sqlite3.connect(database)

# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Create a table to store the stellarator data

cursor.execute("""CREATE TABLE stellarators(id, rbc_0_0, zbs_0_0, rbc_1_0, zbs_1_0, rbc__1_1, zbs__1_1,
    rbc_0_1, zbs_0_1, rbc_1_1, zbs_1_1, Quasisymmetry, Quasiisodynamic, Rotational_Transform, Inverse_Aspect_Ratio,
    Mean_Local_Magnetic_Shear, Vacuum_Magnetic_Well, Maximum_Elongation, Mirror_Ratio, Number_of_Field_Periods_NFP)""")

conn.commit()
conn.close()

