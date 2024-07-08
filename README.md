# PIC-STELLARATOR
The PIC-STELLARATOR repository aims to establish a database for a comprehensive assessment of the conditions under which functional and efficient stellarators can be created and also for posterior analisation with machine learning.

The repository comprises three main folders:
* *data*, which contains the file that serves as a template to the creation of stellarators, a SQL database Python file (to origin the latter) and the actual database.

* *data_base_support*, which contains three files which generate the template of stellarator *VMEC_database.py* and *qi_functions.py*, as well as plotting useful data that comes with the previous files -> *vmecPlot2.py*. Additionally, this folder has the file that creates random stellarators and transports them to the database *auxiliary_functions.py*.

* *data_analysis*: Contains files for interpreting the database. This folder has two subfolders, one for quasi-symmetry and one for quasi-isodynamic properties within the database. For each of these quantities, LightGBM was used to create binary classification and regression models, and LightGBM LSS was used to train a regression model.

## Instalation
Clone the repository and install the necessary packages. In the file *requirements.txt*, you will find all the packages required to run files from the repository.
<pre>
git clone https://github.com/laiarodrigo/PIC-STELLARATOR.git
cd PIC-STELLARATOR
pip install -r requirements.txt
</pre>

## Usage
### Loading Data
Some configurations and their respective results are provided in case the user prefers not to create new stellarators. Otherwise, it is possible to run the <pre> main.py </pre> file. This file will generate database containing the columns:

| Column | Column Name | Description |
|-------------|-------------|-------------|
| `x1`, `x2`, ..., `xN` | The input parameters (RBC's and ZBS' values) | These Fourier coefficients collectively define the stellarator surface, enabling the calculation of the other columns. |
| `y1` | Quasisymmetry | |
| `y2` | Quasiisodynamic | |
| `y3` | Rotational Transform | |
| `y4` | Inverse Aspect Ratio | |
| `y5` | Mean Local Magnetic Shear | |
| `y6` | Vacuum Magnetic Well | |
| `y7` | Maximum Elongation | |
| `y8` | Mirror Ratio | |
| `y9` | Number of Field Periods NFP | |


## Reference Papers

[![ResearchGate](https://img.shields.io/badge/ResearchGate-Magnetic_fields_with_precise_quasisymmetry-brightgreen)](https://www.researchgate.net/publication/353791041_Magnetic_fields_with_precise_quasisymmetry)
[![arXiv](https://img.shields.io/badge/arXiv-2211.09829-brightgreen)](https://arxiv.org/abs/2211.09829)
[![arXiv](https://img.shields.io/badge/arXiv-2006.14881-brightgreen)](https://arxiv.org/abs/2006.14881)


## Acknowledgements
Thanks to Gonçalo Abreu and Rogério Jorge for kickstarting this project.
