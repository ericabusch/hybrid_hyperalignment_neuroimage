# Hybrid hyperalignment: A single high-dimensional model of shared information embedded in cortical patterns of response and functional connectivity
## H2A Pipeline & Analysis Scripts
### Erica Busch, January 2021   

This directory contains the hyperalignment procedures for response hyperalignment, connectivity hyperalignment, and hybrid hyperalignment used in the H2A manuscript. It also contains all scripts used in the final analyses of the Whiplash, Raiders, and Budapest datasets. 


### [dataset]_utils.py  
These files contain all dataset-specific functions and variables. This includes absolute paths for all relevant directories and the subject ids for all analyzed subjects.   

Each file also contains two externally useful functions:   
- `get_train_data`: takes five parameters
	- `side`: can be 'l', 'r', or 'b' indicating which hemisphere data to load
	- `runs`: a list of the run numbers to return (ie if you want to train hyperalignment on 3/4 runs, you can input [1,2,3] or [2,3,4] etc)  
	- `num_subjects`: prepopulated as the total number of subjects in the dataset. Can be changed for debugging.
	- `z` : defaults to True, yes we want to zscore by rows
	- `mask` : defaults to False, depends upon whether the data in the numpy files already masked the medial wall.
and returns a list of PyMVPA datasets of the subjects and concatenated target runs that can be inputted to hyperalignment. 
- `get_test_data`: takes the same parameters as `get_train_data`. Returns a 3-dimensional numpy array instead of a PyMVPA data as our tests do not require the PyMVPA dataset input object.

### searchlight_utils.py
This file contains 1 main variable and several functions: 
- `MASKS`: a dictionary with keys 'l' and 'r' corresponding to the binary fsaverage cortical mask that indicates which nodes on the fsaverage surface are in the medial wall

- `get_node_indices` takes 2 parameters
	- `hemi`: can be 'l','r',or 'b'
	- `surface_res` : defaults to the surface resolution of your data indicated in the dataset_utils file. In our case, this is 10242 nodes per hemisphere before masking the medial wall (ico5). This can be downsampled, for example, to define a sparser surface.
This returns either a single list or a list of lists, depending on if you specify a single hemisphere or both hemispheres, of the indices of the nodes on the fsaverage surface that are not included in the medial wall. 

- `get_freesurfer_surfaces`: takes 1 parameter
	- `hemi`: can be 'l','r', or 'b'.
This loads the freesurfer .white and .pial files for each hemisphere that are saved in the basedir of your project and creates a surface mesh using mvpa2's nibabel support package.

- `compute_searchlights`: takes 1 parameter
	- `hemi`: can be 'l','r', or 'b'.
This function creates a list of lists, where each nested list indicates the indices of the cortical nodes defined on the freesurfer surface that are included in a searchlight of radius SEARCHLIGHT_RADIUS (in dataset_utils file) centered on each node. it uses the surface query engine defined in PyMVPA to use dijkstra's algorithm and define the searchlights and find the nearest nodes within a searchlight.

- `fro_norm_merge` : takes 3 parameters
	- `dss_cnx`: a n_searchlights by n_vertices connectivity matrix
	- `dss_rp`: a n_timepoints by n_vertices data matrix
	- `node_indices` : a list of valid node indices
This function is the key to H2A: it allows us to normalize by the number of rows in each type of data matrix so that neither is contributing more to the derivation of the common space.



# hybrid_hyperalignment_neuroimage
