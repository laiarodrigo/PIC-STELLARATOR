# PIC-STELLARATOR
The PIC-STELLARATOR repository aims to establish a database for a comprehensive assessment of the conditions under which functional and efficient stellarators can be created and also for posterior analisation with machine learning.

The repository comprises three main folders:
* *data*, which contains the file that serves as a template to the creation of stellarators, a SQL database Python file (to origin the latter) and the actual database.

* *data_base_support*, which contains three files which generate the template of stellarator *VMEC_database.py* and *qi_functions.py*, as well as plotting useful data that comes with the previous files -> *vmecPlot2.py*. Additionally, this folder has the file that creates random stellarators and transports them to the database *auxiliary_functions.py*.

* *data_analysis*, that has files to interpret the database.

## Instalation
Clone the repository and install the necessary packages. In the file <pre> requirements.txt </pre>, you will find all the packages required to run files from the repository.
<pre>
git clone https://github.com/laiarodrigo/PIC-STELLARATOR.git
cd PIC-STELLARATOR
pip install -r requirements.txt
</pre>

## Usage
### Loading Data
In the near future, some configurations and their respective results will be provided in case the user prefers not to create new stellarators. Otherwise, it is possible to run the <pre> main.py </pre> file. This file will generate database containing the columns:

| Column Name | Description |
|-------------|-------------|
| `x1`, `x2`, ..., `xN` | The input parameters (RBC's and ZBS' values) |
| `y1` | Quasisymmetry |
| `y2` | Quasiisodynamic |
| `y3` | Rotational Transform |
| `y4` | Inverse Aspect Ratio |
| `y5` | Mean Local Magnetic Shear |
| `y6` | Vacuum Magnetic Well |
| `y7` | Maximum Elongation |
| `y8` | Mirror Ratio |
| `y9` | Number of Field Periods NFP |


## Acknowledgements
Thanks to Gonçalo Abreu and Rogério Jorge for kickstarting this project.
