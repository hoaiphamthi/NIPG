This document provides instructions for the code accompanying the paper 
"New inexact adaptive proximal gradient algorithms for nonconvex composite optimization problems and applications" by Pham Thi Hoai, Nguyen Dang Hao and Jen-Chih Yao.

### Introduction
- The project is modularized into several Python files, separating objective functions (fun_gradobjec.py), proximal operators (prox_operators.py), algorithms (algos_cur.py, algos_iqp.py), and main execution scripts.
- The files main_CUR.py and main_IQPP.py are designed to run multiple problems, sizes, and datasets in a single execution.
- The initial stepsize is determined via line search for algorithms like IPG-ELS and PG-ELS, while the NIPG algorithms utilize a novel adaptive stepsize mechanism.
- We use numpy.random.seed to ensure reproducibility for the generated matrices. Problems with the same size and random seed should always produce identical results.

### Running the Experiments & Data Selection
To run the experiments, execute the corresponding main file (main_CUR.py or main_IQPP.py). Before running, you must configure the data selection directly inside the main scripts:

- For the CUR Problem (main_CUR.py): 
  Real-world datasets are indexed from 1 to 6 (e.g., 1: Colon Tumor, 2: Heart Disease, etc.). To select a specific dataset, modify the range in the main loop. 
  * For example, use "for i in range(1, 2, 1):" to run only dataset 1.
  * Use "for i in range(1, 7, 1):" to run all 6 datasets sequentially.

- For the IQP Problem (main_IQPP.py): 
  Generated matrix sizes are controlled by the dimensions list. You can simply uncomment or add specific dimensions (e.g., (3000, 150)) into the active_dimensions variable to include them in the experiment.

### Obtaining Plots from Presolved Results
- The code automatically packages and saves all experimental results into compressed .npz files located in the saved_data folders. You can load these files later to reproduce the plots without having to re-run the computationally expensive algorithms.

### Experimental results
- The plots generated are automatically saved in the result/CUR/images and result/IQP/images folders.
- The result/CUR/csv_results and result/IQP/csv_results folders contain the summary CSV tables of all the experiments.

### Experimental details
The detailed sizes and datasets for each problem evaluated in the scripts are provided here:
- CUR Problem (Real Datasets): 1. Colon Tumor (62x2000), 2. Heart Disease (303x14), 3. CNS (60x7129), 4. Lung Cancer-Michigan (96x7129), 5. Secom (1567x590), and 6. Cina0 (16033x132).These real data files are located in the "CSV" folder.
- IQP Problem (Generated Matrices): Low-rank (2000x100, 3000x150), Nearly square (500x500), and More columns than rows (100x1000, 250x2500). The random seed is dynamically generated based on matrix dimensions to guarantee consistency.
