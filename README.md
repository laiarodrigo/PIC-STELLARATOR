# PIC-STELLARATOR
The PIC-STELLARATOR repository aims to establish a database for a comprehensive assessment of the conditions under which functional and efficient stellarators can be created.

The repository comprises four main files:

* <pre>importantfuntions.py</pre>: Generates a stellarator, computes the outputs, and creates a CSV where each line corresponds to the configuration of a stellarator.

* <pre>VMEC_database.py</pre>: This is an example file that creates a specific configuration and calculates its outputs.

* <pre>qi_functions.py</pre>:

* <pre>vmecPlot2.py</pre>:     
\\
## Instalation
Clone the repository and install the necessary packages. In the file <pre> requirements.txt </pre>, you will find all the packages required to run files from the repository.
<pre>
git clone https://github.com/laiarodrigo/PIC-STELLARATOR.git
cd PIC-STELLARATOR
pip install -r requirements.txt
</pre>
\\
## Usage
### Loading Data
In the near future, some configurations and their respective results will be provided in case the user prefers not to create new stellarators. Otherwise, it is possible to run the <pre> importantfuntions.py </pre> file. This file will generate a CSV containing the columns:

* <pre> x1 </pre>, <pre> x2 </pre>, ..., <pre> xN </pre>: The input parameters (RBC's and ZBS' values).
* <pre> y1 </pre>: Quasisymmetry
* <pre> y2 </pre>: Quasiisodynamic
* <pre> y3 </pre>: Rotational Transform
* <pre> y4 </pre>: Inverse Aspect Ratio
* <pre> y5 </pre>: Mean Local Magnetic Shear
* <pre> y6 </pre>: Vacuum Magnetic Well
* <pre> y7 </pre>: Maximum Elongation
* <pre> y8 </pre>: Mirror Ratio
* <pre> y9 </pre>: Number of Field Periods NFP

## Acknowledgements
Thanks to Gonçalo Abreu and Rogério Jorge for kickstarting this project.
