#import data_analysis.DataAnalysis as da
import data_base_support.auxiliary_functions as imp
#from simsopt.mhd import Vmec
#from data_base_support.vmecPlot2 import main as vmecPlot2
from pathlib import Path
import data.nfp2.database as db


def main():
    fixed_values = {'rbc_0_0': 1, 'zbs_0_0': 0}

    # Get the current directory of the script
    this_path = Path(__file__).resolve().parent

    # Load the original VMEC input file
    input_vmec_file_original = str(this_path / 'data/nfp2/input.nfp2_qa')

    while True:
        try:
            data = imp.random_search_vmec_input(input_vmec_file_original)
            variable_values = data[1]
            input_data = {**fixed_values, **variable_values}
            stel = data[0]
            output_data = imp.calculate_outputs(stel)

            imp.save_outputs_and_inputs_to_db(output_data, input_data, db.database_file)

        except Exception as e:
            print(f"iteration failed: {e}")
            continue

    #print(imp.run_vmec_simulation_with_plots(69420))

if __name__ == "__main__":
    main()