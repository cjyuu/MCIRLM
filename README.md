<<<<<<< HEAD
﻿#  A representation learning model (MCIRLM) that integrates multi-modal crystal information


## Developers

Cao Jiayu

## Prerequisites

-   Python==3.9
-   pytorch==1.10.1
-   pymatgen== 2023.9.25
-   pandas== 2.1.3
-   numpy==1.26.4
-   matplotlib==3.4.3
-   seaborn==0.13.2
-   scikit-learn==1.0.1
-   tqmd==4.66.1
-   matplotlib==3.4.3


## Usage
### Define a customized dataset

To input crystal structures to MCIRLM, You can create a customized dataset by creating a directory  `data`  with the following files:

-   id_prop.csv: a CSV file with three columns. The first column recodes a unique ID for each crystal, the second column recodes the Chemical formula, and the third column recodes the value of target property.
-   atom_init.json: a JSON file that stores the initialization vector for each element.
-   id.cif: a CIF file that recodes the crystal structure, where ID is the unique ID for the crystal.
-   test.csv
-   val.csv
-   test.csv The structure of the  `data`  should be:
```
root_dir
├── id_prop.csv
├── atom_init.json
├── train.csv
├── val.csv
├── test.csv
├── id0.cif
├── id1.cif
├── ...
```

## Train a MCIRLM model


you can train a MCIRLM model for your customized dataset by:

```
python train.py
```

the configurations are defined in config.yaml.
=======
# MCIRLM
A representation learning model (MCIRLM) that integrates multi-modal crystal information
>>>>>>> 5095305c26bbe71eaf5eeca92fc73255de71823b
