import os
import numpy as np
from simsopt.mhd import Vmec
from vmecPlot2 import main as vmecPlot2
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen
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
            input_data[f'rbc_{n}_{m}'.replace('-', '_')] = {rbc_value}
            input_data[f'zbs_{n}_{m}'.replace('-', '_')] = {zbs_value}

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

def save_outputs_and_inputs_to_csv(outputs, input_vmec_file, output_filename='outputs.csv'):
    # Create a DataFrame with the outputs
    df = pd.DataFrame(outputs, index=[0]).round(10)  # Ajuste o número de casas decimais conforme necessário

    # Create a DataFrame with the inputs
    input_data = read_vmec_input(input_vmec_file)
    df_input = pd.DataFrame([input_data.values()], columns=input_data.keys()).round(3)

    # Concatenate outputs and inputs DataFrames
    final_df = pd.concat([df_input, df], axis=1)
    
    # Save the outputs and inputs to a CSV file with semicolon as separator
    final_df.to_csv(output_filename, index=True, header=True, sep=' ')
    
    return final_df

def random_search_vmec_input(input_vmec_file):
    """
    Perform a random search on VMEC input parameters.
    """
    # Load the original VMEC input file
    stel = Vmec(input_vmec_file, verbose=False)
    surf = stel.boundary

    ## Define how many modes to Use
    max_mode = 1

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
    vmecPlot2(stel.output_file)
    
    
    '''
    ## Change input parameters, degrees of freedom (DOFS)
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # Fix major radius to be the same
    dofs = surf.x
    print()
    print(f'Initial DOFs: {dofs}')
    surf.x=np.array(np.random.uniform(low=-0.5, high=0.5, size=len(stel.x)))
    print(f'New DOFs: {stel.x}')
    print()
    print('teste')
    ## Run initial stellarator and plot
    stel.run()
    vmecPlot2(stel.output_file)
    '''

def main():
    # Original path
    this_path = os.path.dirname(os.path.realpath(__file__))

    # Load the original VMEC input file
    input_vmec_file_original = os.path.join(this_path, 'input.nfp2_QA')
    stel = Vmec(input_vmec_file_original, verbose=False)

    # Random search and save outputs to CSV
    outputs = calculate_outputs(stel)
    random_search_vmec_input(input_vmec_file_original)
    
    
