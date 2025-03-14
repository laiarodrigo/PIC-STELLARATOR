import logging
import data_base_support.auxiliary_functions as imp
from pathlib import Path
import data.nfp2.database as db

from mpi4py import MPI
import sys

def main():
    # fixed_values = {'rbc_0_0': 1, 'zbs_0_0': 0}
    
    # # Get the current directory of the script
    # this_path = Path(__file__).resolve().parent

    # # Load the original VMEC input file
    # input_vmec_file_original = str(this_path / 'data/nfp2/input.nfp2_QA')

    # # Define the number of iterations
    # num_iterations = 20000  # Adjust this number as needed

    # for _ in range(num_iterations):
    #     try:
    #         data = imp.random_search_vmec_input(input_vmec_file_original)
    #         variable_values = data[1]
    #         input_data = {**fixed_values, **variable_values}
    #         stel = data[0]
    #         convergence_status = data[2]
    #         print(f"iteration {_}")
    #         if convergence_status == 1:
    #             output_data = imp.calculate_outputs(stel)
    #             imp.save_outputs_and_inputs_to_db(output_data, input_data, db.database_file, convergence_status)
    #         else:
    #             # Manually set specific keys to None in the default dictionary
    #             default_output_data = {
    #                 'quasisymmetry': 1,
    #                 'quasiisodynamic': 1,
    #                 'rotational_transform': 1,
    #                 'inverse_aspect_ratio': 1,
    #                 'mean_local_magnetic_shear': 1,
    #                 'vacuum_magnetic_well': 1,
    #                 'maximum_elongation': 1,
    #                 'mirror_ratio': 1,
    #                 'number_of_field_periods_nfp': 1
    #             }
    #             imp.save_outputs_and_inputs_to_db(default_output_data, input_data, db.database_file, convergence_status)
    #     except Exception as e:
    #         # Print other exceptions normally
    #         print(f"Iteration failed: {e}")
    #         continue
    #imp.run_vmec_simulation_with_plots(10023083)
    #imp.run_vmec_simulation_with_plots(1174872)
    #imp.run_vmec_simulation_with_plots(11298784)
    #imp.run_vmec_simulation_with_plots(244508)
    #imp.run_vmec_simulation_with_plots(7091918)
    ################
    #imp.run_vmec_simulation_with_plots(4918980)
    imp.run_vmec_simulation_with_plots(4796644)
    #imp.run_vmec_simulation_with_plots(11232037)
    #imp.run_vmec_simulation_with_plots(3062344)

if __name__ == "__main__":
    main()