import os
import other_files.importantfunctions as imp
from simsopt.mhd import Vmec
from other_files.vmecPlot2 import main as vmecPlot2
from pathlib import Path
import database as db


def main():
    fixed_values = {'rbc_0_0': 1, 'zbs_0_0': 0}

    # Get the current directory of the script
    this_path = Path(__file__).resolve().parent

    # Load the original VMEC input file
    input_vmec_file_original = str(this_path / 'data/nfp2/input.nfp2_QA')

    for i in range(1, 3000):
        print(i)
        try:
            data = imp.random_search_vmec_input(input_vmec_file_original)
            variable_values = data[1]     
            input_data = {**fixed_values, **variable_values}
            stel = data[0]
            output_data = imp.calculate_outputs(stel)
            
            imp.save_outputs_and_inputs_to_db(output_data, input_data, db.database_file)
            
        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            continue

if __name__ == "__main__":
    main()