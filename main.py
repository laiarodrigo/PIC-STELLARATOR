import logging
import data_base_support.auxiliary_functions as imp
from pathlib import Path
import data.nfp2.database as db

from mpi4py import MPI
import sys

def main():
    fixed_values = {'rbc_0_0': 1, 'zbs_0_0': 0}
    comm = MPI.COMM_WORLD

    # Get the current directory of the script
    this_path = Path(__file__).resolve().parent

    # Load the original VMEC input file
    input_vmec_file_original = str(this_path / 'data/nfp2/input.nfp2_QA')

    while True:
        try:
            data = imp.random_search_vmec_input(input_vmec_file_original)
            variable_values = data[1]
            input_data = {**fixed_values, **variable_values}
            stel = data[0]
            output_data = imp.calculate_outputs(stel)

            imp.save_outputs_and_inputs_to_db(output_data, input_data, db.database_file)

        except MPI.Exception as e:
            # Print rank and error message
            rank = comm.Get_rank()
            mpi_version = MPI.__version__
            mpi_vendor = MPI.get_vendor()
            mpi_ident = MPI.get_ident()
            mpi_revision = MPI.get_revision()
            mpi_date = MPI.get_date()
            mpi_version_str = f"mpi4py v{mpi_version}, package: {mpi_vendor}, ident: {mpi_ident}, repo rev: {mpi_revision}, {mpi_date}"
            sys.stderr.write(f"Rank {rank}: MPI error occurred: {e}\n")
            sys.stderr.write(f"MPI version: {mpi_version_str}\n")
        except Exception as e:
            # Print other exceptions normally
            print(f"iteration failed: {e}")
        continue

if __name__ == "__main__":
    main()
