import database as db
import sqlite3
import numpy as np
from simsopt.mhd import Vmec
from other_files.vmecPlot2 import main as vmecPlot2
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from other_files.qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen
import re
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

def save_outputs_and_inputs_to_db(outputs: dict, input_data: dict, database_file):
    # # Create a DataFrame with the outputs
    # df = pd.DataFrame(outputs, index=[0]).round(10)  # Ajuste o número de casas decimais conforme necessário

    #df_input = pd.DataFrame([input_data.values()], columns=input_data.keys()).round(3)

    # Concatenate outputs and inputs DataFrames
    #final_df = pd.concat([df_input, df], axis=1)
    final_dict = {**input_data, **outputs}
    
    values_tuple= tuple(final_dict.values())
  
    conn = sqlite3.connect(database_file)
    new_cursor = conn.cursor()

    # Insert data into the database
    new_cursor.execute("""INSERT INTO stellarators 
                        (rbc_0_0, zbs_0_0, rbc_1_0, rbc_m1_1, rbc_0_1, rbc_1_1, zbs_1_0, zbs_m1_1,
                        zbs_0_1, zbs_1_1, quasisymmetry, quasiisodynamic,
                        rotational_transform, inverse_aspect_ratio, mean_local_magnetic_shear,
                        vacuum_magnetic_well, maximum_elongation, mirror_ratio,
                        number_of_field_periods_nfp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        values_tuple)

    # Commit the transaction
    conn.commit()

    # Close the connection
    conn.close()
    print(values_tuple)

def random_search_vmec_input(input_vmec_file):
    """
    Perform a random search on VMEC input parameters.
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
    surf.x=np.array(np.random.uniform(low=-0.1, high=0.1, size=len(stel.x)))
    print(f'New DOFs: {stel.x}')
    print()
    ## Run initial stellarator and plot
    stel.run()
    vmecPlot2(stel.output_file)
    
    input_data = {}
    
    for i in range(len(surf.x)):
        input_data[f'dof_{i}'] = surf.x[i]

    return [stel, input_data]
    

    
    
