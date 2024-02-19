import os
import numpy as np
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
from other_files.vmecPlot2 import main as vmecPlot2
from other_files.qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen

this_path = os.path.dirname(os.path.realpath(__file__))
input_vmec_file = os.path.join(this_path, 'input.nfp2_QA')
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

## Check output parameters
qs = np.sum(QuasisymmetryRatioResidual(stel, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).residuals()**2)
qi = np.sum(QuasiIsodynamicResidual(stel,[1/16,5/16])**2)
iota = stel.mean_iota()
epsilon = 1/stel.aspect()
shear = stel.mean_shear()
well = stel.vacuum_well()
elongation = np.max(MaxElongationPen(stel))
mirror = MirrorRatioPen(stel)
nfp = stel.wout.nfp
print()
print("Output Parameters:")
print(f' Quasisymmetry: {qs}')
print(f' Quasiisodynamic: {qi}')
print(f' Rotational Transform: {iota}')
print(f' Inverse Aspect Ratio: {epsilon}')
print(f' Mean local magnetic Shear: {shear}')
print(f' Vacuum Magnetic Well: {well}')
print(f' Maximum Elongation: {elongation}')
print(f' Mirror Ratio: {mirror}')
print(f' Number of Field Periods NFP: {nfp}')
