## Hybrid hyperalignment: A single high-dimensional model of shared information embedded in cortical patterns of response and functional connectivity
## H2A Pipeline & Analysis Scripts

### Erica Busch, March 2021   

This directory contains the hyperalignment procedures for response hyperalignment, connectivity hyperalignment, and hybrid hyperalignment used in the H2A manuscript. It also contains all scripts used in the final analyses of the Whiplash, Raiders, and Budapest datasets. 

For more information, check out our [paper](https://www.biorxiv.org/content/10.1101/2020.11.25.398883v1), in press at NeuroImage. 

### Scripts & Functions
- `[dataset]_utils.py` : One for each of the three datasets used here (Raiders, Whiplash, & Budapest). These scripts include relevant paths, subject IDs, and dataset-specific information, as well as functions for accessing each dataset.   

- `HA_prep_functions.py` : An amalgamation of functions used to prepare input data for hyperalignment.

- `drive_hyperalignment_cross_validation.py` : This script drives the training of the three flavors of hyperalignment models tested in this paper in a leave-one-run-out cross-validation scheme. This script takes 2 (optionally 3) command-line arguments - type of hyperalignment model to train, the dataset, and the held-out run number. If 2 arguments, it runs through all train-test combos. After training the HA model on the training datasets, it derives mappers for each subject's data into the trained common space, saves those mappers with `save_transformations` and transforms and saves the test data in the common model space with `save_transformed_data`. It then calls `run_benchmarks` on the transformed test data. 

- `benchmarks.py` : includes source code for vertex-wise intersubject correlations, connectivity profile intersubject correlations, and sliding window between-subject multivariate pattern classifications.

- `run_anatomical_benchmarks.py` : drives the control analyses (running each of the benchmarks on the anatomically aligned data, pre-hyperalignment)

- `hybrid_hyperalignment.py` : the class defining the hybrid hyperalignment method and source code.