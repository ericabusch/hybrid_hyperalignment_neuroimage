# whiplash_utils.py
# utils file specific to whiplash dataset; all dataset-specific code will be kept here
# erica busch, 2020
import os,glob
import pandas as pd
import numpy as np
from scipy.stats import zscore
from mvpa2.datasets.base import Dataset

basedir = '/dartfs/rc/lab/D/DBIC/DBIC/f002d44/h2a'
connhyper_dir = os.path.join(basedir, 'connhyper','whiplash')
resphyper_dir = os.path.join(basedir, 'response_hyper','whiplash')
datadir = os.path.join(basedir, 'data','whiplash')
orig_datadir = os.path.join(datadir, 'whiplash')
connectome_dir = os.path.join(datadir, 'connectomes')
results_dir = os.path.join(basedir, 'results','whiplash')
h2a_dir = os.path.join(basedir, 'iterative_hyper')

for dn in [connhyper_dir, resphyper_dir, h2a_dir, results_dir, iterative_HA_dir]:
	if not os.path.isdir(dn):
		os.makedirs(dn)
		print('made '+str(dn))

subj_df = pd.read_csv(os.path.join(datadir,'whiplash_subjects.csv'))['subject_id']
subjects = [s.split('sub-sid')[1] for s in sorted(list(subj_df))]
num_subjects=len(subjects)

SURFACE_RESOLUTION = 10242
SEARCHLIGHT_RADIUS = 13 
HYPERALIGNMENT_RADIUS = 20
TOT_TRs = 1770
TOT_RUNS = 4
NNODES_LH = 9372
MASKS = {'lh':np.load(basedir+'/fsaverage_lh_mask.npy')[:SURFACE_RESOLUTION], 
         'rh':np.load(basedir+'/fsaverage_rh_mask.npy')[:SURFACE_RESOLUTION]}

midstr = '_ses-3-task-movie_run-02'
endstr = '_space-fsaverage_hemi-'

# this dataset is collected all in one run so we have to manually divide the session into 4 runs.
# we're going to make each of these 'runs' 443 TRs long
n = int(round(TOT_TRs/TOT_RUNS)+1)
TR_run_chunks = [np.arange(TOT_TRs)[i:i + n] for i in range(0, TOT_TRs, n)] 

def get_RHA_data(runs):
    dss=[]
    for run in runs:
        ds_lh = np.load(os.path.join(resphyper_dir, 'data','fold_{0}'.format(run), 'dss_lh.npy'))
        ds_rh = np.load(os.path.join(resphyper_dir,'data', 'fold_{0}'.format(run), 'dss_rh.npy'))
        dss.append(np.concatenate((ds_lh, ds_rh),axis=2))
    return np.array(dss)

# runs indicates which runs we want to return.
# this will be useful for datafolding.
def get_train_data(side, runs, num_subjects=num_subjects, z=True, mask=True):
	dss = []
	TR_train = np.concatenate([TR_run_chunks[i-1] for i in runs])
	for subj in subjects[:num_subjects]: 
		data = _get_whiplash_data(subj, side, z, mask)
		data = data[TR_train,:]
		ds = Dataset(data)
		dss.append(ds)
	return dss

# specific formatting for budapets data; only gets called internally.
def _get_whiplash_data(subject, side, z, mask):
	LR = side.lower()
	if LR == 'b':
		return np.hstack([_get_whiplash_data(subject, 'lh', z, mask),
		 _get_whiplash_data(subject, 'rh', z, mask)])
	a = orig_datadir+'/*{s}_{LR}*.npy'.format(s=subject,LR=LR.lower())
	fn = glob.glob(a)[0]
	d = np.nan_to_num(np.load(fn))
	if mask:
		d = d[:,MASKS[side]]
	if z:
		d = zscore(d)
	return d

# dont want to return this as a pymvpa dataset; takes too long & is unnecessary
def get_test_data(side, runs, num_subjects=29, z=True, mask=True):
	dss=[]
	TR_test = np.concatenate([TR_run_chunks[i-1] for i in runs])
	for subj in subjects[:num_subjects]:
		ds = _get_whiplash_data(subj, side, z, mask)
		ds = ds[TR_test,:]
		dss.append(ds)
	return np.array(dss)

















