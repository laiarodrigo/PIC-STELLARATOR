import sqlite3
from pathlib import Path

combined_db_path = Path('nfp2_combined.db')

def find_stellarators_with_low_sum(db_path, weight_quasiisodynamic=10):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to find the 30 stellarators with the lowest summed value
        # (weighted quasiisodynamic + 1/inverse_aspect_ratio)
        query = f"""
            SELECT id, quasiisodynamic, inverse_aspect_ratio, 
                   ({weight_quasiisodynamic} * quasiisodynamic + (1.0 / inverse_aspect_ratio)) AS sum_value
            FROM stellarators_combined
            WHERE convergence = 1
                AND quasiisodynamic IS NOT NULL
                AND inverse_aspect_ratio IS NOT NULL
            ORDER BY sum_value ASC
            LIMIT 30;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        conn.close()

        if rows:
            # Print the IDs, quasiisodynamic, inverse_aspect_ratio, and the calculated sum_value
            print(f"{'ID':<10}{'Quasiisodynamic':<20}{'Inverse Aspect Ratio':<25}{'Sum Value':<20}")
            print("-" * 75)
            for row in rows:
                print(f"{row[0]:<10}{row[1]:<20}{row[2]:<25}{row[3]:<20.4f}")

            # Calculate the total sum of quasiisodynamic and 1/inverse_aspect_ratio for the selected stellarators
            total_quasiisodynamic = sum(row[1] for row in rows)
            total_inverse_aspect_ratio = sum(1.0 / row[2] for row in rows)
            total_sum = sum(row[3] for row in rows)

            print("\nTotal values for the selected stellarators:")
            print(f"Total Quasiisodynamic: {total_quasiisodynamic:.4f}")
            print(f"Total 1/Inverse Aspect Ratio: {total_inverse_aspect_ratio:.4f}")
            print(f"Total Sum Value: {total_sum:.4f}")
        else:
            print("No stellarators found matching the criteria.")

    except sqlite3.DatabaseError as e:
        print(f"Error accessing database: {e}")
    except ValueError as e:
        print(e)

# Call the function and print the result
find_stellarators_with_low_sum(combined_db_path)
