# budapest_utils.py
# utils file specific to your dataset; all dataset-specific code will be kept here
# erica busch, 2020

import numpy as np
from scipy.stats import zscore
import os
from mvpa2.datasets.base import Dataset

basedir = '/dartfs/rc/lab/D/DBIC/DBIC/f002d44/h2a'
home_dir = '/dartfs-hpc/rc/home/4/f002d44/h2a'
connhyper_dir = os.path.join(basedir, 'connhyper','budapest')
resphyper_dir = os.path.join(basedir, 'response_hyper','budapest')
h2a_dir = os.path.join(basedir, 'mixed_hyper','budapest')
datadir = os.path.join(basedir, 'data','budapest')
orig_datadir = os.path.join(datadir, 'original')
connectome_dir = os.path.join(datadir, 'connectomes')
results_dir = os.path.join(basedir, 'results','budapest')
iterative_HA_dir = os.path.join(basedir, 'iterative_hyper')

sub_nums = [5, 7, 9, 10, 13, 20, 21, 24, 29, 34, 52, 114, 120, 134, 142, 
278, 416, 499, 522, 535, 560]
subjects =  ['{:0>6}'.format(subid) for subid in sub_nums]
SURFACE_RESOLUTION = 10242
SEARCHLIGHT_RADIUS = 13 
HYPERALIGNMENT_RADIUS = 20
TOT_RUNS= 5
NNODES_LH=9372

midstr = '_ses-budapest_task-movie_run-'
endstr = '_space-fsaverage-icoorder5_hemi-'

def get_RHA_data(runs, num_subjects=21, training=False):
    ds_lh = np.load(os.path.join(resphyper_dir, 'fold_{0}'.format(runs[0]),'data', 'dss_lh.npy'))
    ds_rh = np.load(os.path.join(resphyper_dir, 'fold_{0}'.format(runs[0]),'data', 'dss_rh.npy'))
    dss= np.concatenate((ds_lh,ds_rh),axis=2)
    if len(runs)>1:
        for run in runs[1:]:
            ds_lh = np.load(os.path.join(resphyper_dir, 'fold_{0}'.format(run),'data', 'dss_lh.npy'))
            ds_rh = np.load(os.path.join(resphyper_dir, 'fold_{0}'.format(run),'data', 'dss_rh.npy'))
            arr = np.concatenate((ds_lh, ds_rh),axis=2)
            dss=np.concatenate((dss,arr),axis=1)
        dss = format_for_training(dss,num_subjects)
    return dss

def format_for_training(dss, num_subjects):
    dss_formatted = []
    for ds,subj in zip(dss,subjects[:num_subjects]):
        data = zscore(ds, axis=0)
        dss_formatted.append(Dataset(data))
    return dss_formatted
        
# runs indicates which runs we want to return.
# this will be useful for datafolding.
def get_train_data(side, runs, num_subjects=21, z=True, mask=False):
	dss = []
	for subj in subjects[:num_subjects]:
		data = _get_budapest_data(subj, side.upper(), runs, z, mask)
		ds = Dataset(data)
		idx = np.where(np.logical_not(np.all(ds.samples == 0, axis=0)))[0]
		ds = ds[:, idx]
		dss.append(ds)
	return dss

# specific formatting for budapets data; only gets called internally.
def _get_budapest_data(subject, side, runs, z, mask):
	LR = side.upper()
	run_list = ['{:0>2}'.format(r) for r in runs]
	if LR == 'B':
		return np.hstack([_get_budapest_data(subject, 'L', runs, z, mask),
		 _get_budapest_data(subject, 'R', runs, z, mask)])
	fns = ['{d}/sub-sid{s}{m}{r}{e}{LR}.func.npy'.format(d=orig_datadir, s=subject, m=midstr, 
		r=i,e=endstr,LR=LR.upper()) for i in run_list]
	ds = [zscore(np.load(fn),axis=0) for fn in fns]
	dss = np.concatenate(ds,axis=0)
	return dss

# dont want to return this as a pymvpa dataset; takes too long & is unnecessary
def get_test_data(side, runs, num_subjects=21, z=True, mask=False):
	dss=[]
	for subj in subjects[:num_subjects]:
		ds = _get_budapest_data(subj, side.upper(), runs, z, mask)
		dss.append(ds)
	return np.array(dss)
