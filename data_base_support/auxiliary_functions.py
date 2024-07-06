import sys
from pathlib import Path

# Add the parent directory of `data` to the Python path
parent_dir_of_data = str(Path(__file__).resolve().parent.parent)
if parent_dir_of_data not in sys.path:
    sys.path.append(parent_dir_of_data)

import data.nfp2.database as db
import sqlite3
import numpy as np
from simsopt.mhd import Vmec
from data_base_support.vmecPlot2 import main as vmecPlot2
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from data_base_support.qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen
import re
from pathlib import Path
import pandas as pd

# Vmec reader, extract RBC and ZBS values from input file
def read_vmec_input(file_path):
    input_data = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Skip comments and empty lines
        if line.strip().startswith('!') or not line.strip():
            continue

        # Match lines with RBC and ZBS values
        match = re.match(r'RBC\(\s*(-?\d+)\s*,\s*(-?\d+)\)\s*=\s*([eE0-9\+\-\.]+),\s*ZBS\(\s*(-?\d+)\s*,\s*(-?\d+)\)\s*=\s*([eE0-9\+\-\.]+)', line)
        if match:
            n = int(match.group(1))
            m = int(match.group(2))
            rbc_value = float(match.group(3))
            zbs_value = float(match.group(6))
            input_data[f'rbc_{n}_{m}'.replace('-', 'm')] = rbc_value
            input_data[f'zbs_{n}_{m}'.replace('-', 'm')] = zbs_value

    return input_data 
   
def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key}: {value}')

def calculate_outputs(stel: Vmec):
    # Quasisymmetry Ratio Residual
    qs = np.sum(QuasisymmetryRatioResidual(stel, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).residuals()**2)

    # Quasi-Isodynamic Residual
    qi = np.sum(QuasiIsodynamicResidual(stel,[1/16,5/16])**2)

    # Rotational Transform
    iota = stel.mean_iota()

    # Inverse Aspect Ratio
    epsilon = 1 / stel.aspect()

    # Mean Local Magnetic Shear
    shear = stel.mean_shear()

    # Vacuum Magnetic Well
    well = stel.vacuum_well()

    # Maximum Elongation
    elongation = np.max(MaxElongationPen(stel))

    # Mirror Ratio
    mirror = MirrorRatioPen(stel)

    # Number of Field Periods NFP
    nfp = stel.wout.nfp

    outputs = {
        'Quasisymmetry': qs,
        'Quasiisodynamic': qi,
        'Rotational_Transform': iota,
        'Inverse_Aspect_Ratio': epsilon,
        'Mean_Local_Magnetic_Shear': shear,
        'Vacuum_Magnetic_Well': well,
        'Maximum_Elongation': elongation,
        'Mirror_Ratio': mirror,
        'Number_of_Field_Periods_NFP': nfp
    }

    return outputs

def save_outputs_and_inputs_to_db(outputs: dict, input_data: dict, database_file, convergence_status):
    # Combine input_data and outputs
    final_data = {**input_data, **outputs, 'convergence': convergence_status}

    # If convergence_status is 0, set specific keys to None
    if convergence_status == 0:
        final_data.update({
            'quasisymmetry': None,
            'quasiisodynamic': None,
            'rotational_transform': None,
            'inverse_aspect_ratio': None,
            'mean_local_magnetic_shear': None,
            'vacuum_magnetic_well': None,
            'maximum_elongation': None,
            'mirror_ratio': None,
            'number_of_field_periods_nfp': None
        })

    print(final_data)

    # Ensure integer fields are rounded and converted to int
    if 'Number_of_Field_Periods_NFP' in final_data and final_data['Number_of_Field_Periods_NFP'] is not None:
        final_data['Number_of_Field_Periods_NFP'] = int(final_data['Number_of_Field_Periods_NFP'])

    # Create a tuple of values to be inserted into the database
    values_tuple = tuple(final_data.values())

    # Connect to the database
    conn = sqlite3.connect(database_file)

    # Create a cursor object
    cursor = conn.cursor()

    # Insert data into the database
    cursor.execute("""INSERT INTO examples
                      (rbc_0_0, zbs_0_0, rbc_1_0, rbc_m1_1, rbc_0_1, rbc_1_1, zbs_1_0, zbs_m1_1,
                      zbs_0_1, zbs_1_1, quasisymmetry, quasiisodynamic,
                      rotational_transform, inverse_aspect_ratio, mean_local_magnetic_shear,
                      vacuum_magnetic_well, maximum_elongation, mirror_ratio,
                      number_of_field_periods_nfp, convergence)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      values_tuple)

    # Commit the transaction
    conn.commit()

    # Close the connection
    conn.close()
    print("Data saved to database.")



def random_search_vmec_input(input_vmec_file):
    """
    Perform a random search on VMEC input parameters.
    Return the VMEC object and a boolean indicating whether the MHD differential equation converges or not.
    """
    # Load the original VMEC input file
    stel = Vmec(input_vmec_file, verbose=False)
    surf = stel.boundary

    ## Define how many modes to Use
    max_mode = 1

    '''
    ## Change input parameters, degrees of freedom (DOFS)
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # Fix major radius to be the same
    dofs = surf.x
    print()
    print(f'Initial DOFs: {dofs}')
    surf.x=np.array([0.07 ,  0.016,  0.18 ,  0.01 , -0.13 ,  0.01 ,  0.2  ,  0.06])
    print(f'New DOFs: {stel.x}')
    print()
    ## Run initial stellarator and plot
    stel.indata.mpol = max_mode + 3
    stel.indata.ntor = max_mode + 3
    stel.run()
    vmecPlot2(stel.output_file)'''
    
    ## Change input parameters, degrees of freedom (DOFS)

    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # Fix major radius to be the same
    dofs = surf.x
    print()
    print(f'Initial DOFs: {dofs}')
    surf.x = np.array(np.random.uniform(low=-0.1, high=0.1, size=len(stel.x)))
    print(f'New DOFs: {stel.x}')
    print()
    ## Run initial stellarator and plot
    try:
        stel.run()
        convergence_status = 1  # If it runs without errors, consider it converged
    except Exception as e:
        print(f"Error algo aconteceu occurred during VMEC run: {e}")
        convergence_status = 0  # If an error occurs, consider it not converged
    
    input_data = {}
    
    for i in range(len(surf.x)):
        input_data[f'dof_{i}'] = surf.x[i]

    return [stel, input_data, convergence_status]
    
def run_vmec_simulation_with_plots(record_id):
    
    this_path = Path(__file__).resolve().parent.parent
    print(this_path)


    # Construct the path to the database file
    db_path =  this_path / 'data' / 'nfp2' / 'nfp2_combined.db' #erro mudar path
    print(db_path)

    # Connect to the SQLite database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Retrieve the specific record from the database
    cursor.execute("SELECT * FROM stellarators_combined WHERE id=?", (record_id,))
    record = cursor.fetchone()
    conn.close()

    if record is None:
        print("Record not found.")
        return

    # Load the original VMEC input file
    input_vmec_file_original = str(this_path / 'data_base_support/input.nfp2_QA')
    stel = Vmec(input_vmec_file_original, verbose=False)
    surf = stel.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False) 
    surf.fix("rc(0,0)") # Fix major radius to be the same
    surf.x=np.concatenate((record[2:6], record[7:11]),axis=0)
    outputs = record[12:]
    stel.run()
    ## Run initial stellarator and plot
    vmecPlot2(stel.output_file)
    return outputs