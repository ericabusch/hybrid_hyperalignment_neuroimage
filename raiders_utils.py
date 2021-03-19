# raiders_utils.py
# utils file specific to raiders_utils; all dataset-specific code will be kept here
# erica busch, 2020

import numpy as np
from scipy.stats import zscore
import os
from mvpa2.datasets.base import Dataset

basedir = '/dartfs/rc/lab/D/DBIC/DBIC/f002d44/h2a'
home_dir = '/dartfs-hpc/rc/home/4/f002d44/h2a'
connhyper_dir = os.path.join(basedir, 'connhyper','raiders')
resphyper_dir = os.path.join(basedir, 'response_hyper','raiders')
h2a_dir = os.path.join(basedir, 'mixed_hyper','raiders')
datadir = os.path.join(basedir, 'data','raiders')
orig_datadir = os.path.join(datadir, 'original')
connectome_dir = os.path.join(datadir, 'connectomes')
results_dir = os.path.join(basedir, 'results','raiders')
iterative_HA_dir = os.path.join(basedir, 'iterative_hyper')

sub_nums = [5, 7, 9, 10, 12, 13, 20, 21, 24, 29, 34, 52, 102, 114, 120, 134, 142, 
278, 416, 433, 499, 522, 535]
subjects =  ['{:0>6}'.format(subid) for subid in sub_nums]
SURFACE_RESOLUTION = 10242
SEARCHLIGHT_RADIUS = 13 
HYPERALIGNMENT_RADIUS = 20
TOT_RUNS= 4
MASKS = {'l':np.load(basedir+'/fsaverage_lh_mask.npy')[:SURFACE_RESOLUTION], 
         'r':np.load(basedir+'/fsaverage_rh_mask.npy')[:SURFACE_RESOLUTION]}
NNODES_LH = 9372

def get_RHA_data(runs, num_subjects=21):
    dss=[]
    for run in runs:
        ds_lh = np.load(os.path.join(resphyper_dir, 'fold_{0}'.format(run), 'dss_lh.npy'))
        ds_rh = np.load(os.path.join(resphyper_dir, 'fold_{0}'.format(run), 'dss_rh.npy'))
        dss.append(np.concatenate((ds_lh, ds_rh),axis=2))
    return np.array(dss)

# runs indicates which runs we want to return.
# this will be useful for datafolding.
def get_train_data(side, runs, num_subjects=23, z=True, mask=True):
	dss = []
	for subj in subjects[:num_subjects]:
		data = _get_raiders_data(subj, side.upper(), runs, z, mask)
		ds = Dataset(data)
		idx = np.where(np.logical_not(np.all(ds.samples == 0, axis=0)))[0]
		ds = ds[:, idx]
		dss.append(ds)
	return dss

# specific formatting for raiders data; only gets called internally.
def _get_raiders_data(subject, side, runs, z, mask):
	side = side.lower()
	run_list = ['{:0>2}'.format(r) for r in runs]
	if side == 'b':
		return np.hstack([_get_raiders_data(subject, 'l', runs, z, mask),
		 _get_raiders_data(subject, 'r', runs, z, mask)])
	fns = ['{d}/sid{s}_{h}h_movie_{r}.npy'.format(d=orig_datadir, s=subject, h=side, r=i) for i in run_list]
	ds = []
	for fn in fns:
		d = np.load(fn)
		if mask:
			d = d[:,MASKS[side]]
		if z:
			d = zscore(d,axis=0)
		ds.append(d)        
	dss = np.concatenate(ds,axis=0)
	return dss

# dont want to return this as a pymvpa dataset; takes too long & is unnecessary
def get_test_data(side, runs, num_subjects=23, z=True, mask=True):
	dss=[]
	for subj in subjects[:num_subjects]:
		ds = _get_raiders_data(subj, side, runs, z, mask)
		dss.append(ds)
	return np.array(dss)














