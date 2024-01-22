import sqlite3
from importantfunctions import main

# Replace 'nome_do_banco.db' with the name of your SQLite database
database = 'stellarators.db'

# Connect to the database (create if not exists)
conn = sqlite3.connect(database)

# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Call the main function to get the columns
config = main()

# Get the list of column names
column_names = config.columns.tolist()

# Define data types for each column
column_types = [f'{column} REAL' for column in column_names]

# Create a string for the columns in the CREATE TABLE statement
columns_definition = ', '.join(column_types)

# Create the CREATE TABLE statement
create_table = f'''
    CREATE TABLE IF NOT EXISTS stellarators (
        id INTEGER PRIMARY KEY,
        {columns_definition}
    )
'''

# Execute the query to create the table
cursor.execute(create_table)
conn.commit()
print(create_table)

'''
# Now, you can create a DataFrame with the columns that are always present
columns_to_add = {
    'Quasisymmetry': 'REAL',
    'Quasiisodynamic': 'REAL',
    'Rotational_Transform': 'REAL',
    'Inverse_Aspect_Ratio': 'REAL',
    'Mean_Local_Magnetic_Shear': 'REAL',
    'Vacuum_Magnetic_Well': 'REAL',
    'Maximum_Elongation': 'REAL',
    'Mirror_Ratio': 'REAL',
    'Number_of_Field_Periods_NFP': 'REAL'
}


# Add the columns to the DataFrame
alter_query = f"ALTER TABLE stellarators ADD COLUMN {columns_to_add.keys()} {columns_to_add.values()}"

cursor.execute(alter_query)
conn.commit()
conn.close()'''

