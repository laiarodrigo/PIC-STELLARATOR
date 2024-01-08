import os
import numpy as np
from simsopt.mhd import Vmec
from vmecPlot2 import main as vmecPlot2
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen
import re
import copy

def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key}: {value}')

# Função para ler os RBCs e ZBSs de um arquivo VMEC
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
            input_data[(n, m)] = {'RBC': rbc_value, 'ZBS': zbs_value}

    return input_data 
   
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
        'Rotational Transform': iota,
        'Inverse Aspect Ratio': epsilon,
        'Mean Local Magnetic Shear': shear,
        'Vacuum Magnetic Well': well,
        'Maximum Elongation': elongation,
        'Mirror Ratio': mirror,
        'Number of Field Periods NFP': nfp
    }

    return outputs

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
    surf.x=np.array(np.random.uniform(low=-0.5, high=0.5, size=len(stel.x)))
    print(f'New DOFs: {stel.x}')
    print()
    print('teste')
    ## Run initial stellarator and plot
    stel.run()
    vmecPlot2(stel.output_file)
    
# Diretório atual do script
this_path = os.path.dirname(os.path.realpath(__file__))

# Caminho do arquivo VMEC original
input_vmec_file_original = os.path.join(this_path, 'input.nfp2_QA')
stel = Vmec(input_vmec_file_original, verbose=False)

print(calculate_outputs(Vmec(input_vmec_file_original, verbose=False)))
random_search_vmec_input(input_vmec_file_original)

