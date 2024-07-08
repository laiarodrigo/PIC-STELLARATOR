# PIC-STELLARATOR
The PIC-STELLARATOR repository aims to establish a database for a comprehensive assessment of the conditions under which functional and efficient stellarators can be created and also for posterior analisation with machine learning.

The repository comprises three main folders:
* <pre>data</pre> ,
which contains the file that serves as a template to the creation of stellarators, a sql database python file (to origin the latter) and the the actual database. 

* <pre>data_base_support</pre> 
which contains three files which generate the template of stellarator <pre>VMEC_database.py</pre> and <pre>qi_functions.py</pre>, as well as plotting useful data that comes with the previous files -> <pre>vmecPlot2.py</pre> Additionaly, this folder has the file creates random stellarators and transports them to the database <pre>auxiliary_functions.py</pre>

* <pre>data_analysis</pre> That has files to interpret the database.
     
\
## Instalation
Clone the repository and install the necessary packages. In the file <pre> requirements.txt </pre>, you will find all the packages required to run files from the repository.
<pre>
git clone https://github.com/laiarodrigo/PIC-STELLARATOR.git
cd PIC-STELLARATOR
pip install -r requirements.txt
</pre>
\
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
