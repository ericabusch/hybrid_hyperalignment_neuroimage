# drive_hyperalignment_cross_validation.py
# erica busch, 2020
# this script takes 2 (optionally 3) command line arguments and runs leave-one-run-out 
# cross validation on the hyperalignment training.
# 1) type of hyperalignment [either RHA, H2A, or CHA]
# 2) dataset to use [either raiders, whiplash, or budapest]
# 3) the run to test on (which means you're training the HA model on the other runs)

import os, sys, itertools
import numpy as np
from scipy.sparse import load_npz, save_npz
from mvpa2.base.hdf5 import h5save, h5load 
from scipy.stats import zscore
import HA_prep_functions as prep
import hybrid_hyperalignment as h2a
from benchmarks import searchlight_timepoint_clf, vertex_isc, dense_connectivity_profile_isc

os.environ['TMPDIR'] = '/dartfs-hpc/scratch/f002d44/temp'
os.environ['TEMP'] = '/dartfs-hpc/scratch/f002d44/temp'
os.environ['TMP'] = '/dartfs-hpc/scratch/f002d44/temp'
N_LH_NODES_MASKED = 9372
N_JOBS=16
N_BLOCKS=128
TOTAL_NODES=10242
SPARSE_NODES=642
HYPERALIGNMENT_RADIUS=20

def save_transformations(transformations, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    h5save(outdir+'/all_subjects_mappers.hdf5', transformations)
    for T, s in zip(transformations, utils.subjects):
        save_npz(outdir+"/subj{}_mapper.npz".format(s), T.proj)
    
# apply the HA transformations to the testing data, splits into hemispheres, saves
def save_transformed_data(transformations, data, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    dss_lh,dss_rh=[],[]
    for T, d, sub in zip(transformations, data, utils.subjects):
        aligned = np.nan_to_num(zscore((np.asmatrix(ds)*T).A, axis=0))
        ar, al = aligned[:,N_LH_NODES_MASKED:], aligned[:,:N_LH_NODES_MASKED]
        dss_rh.append(ar)
        dss_lh.append(al)
    np.save(outdir+'/dss_lh.npy', np.array(dss_lh))
    np.save(outdir+'/dss_rh.npy', np.array(dss_rh))
    print('saved at {}'.format(outdir))
    
# runs benchmarks and saves
def run_benchmarks(fold_basedir):
    results_dir, data_dir = os.path.join(fold_basedir, 'results'), os.path.join(fold_basedir, 'data')
    dss_lh, dss_rh = np.load(data_dir+'/dss_lh.npy'), np.load(data_dir+'/dss_rh.npy')
    lh_res = vertex_isc(dss_lh)
    rh_res = vertex_isc(dss_rh)
    np.save(os.path.join(results_dir, 'vertex_isc_lh.npy',lh_res))
    np.save(os.path.join(results_dir, 'vertex_isc_rh.npy',rh_res))
    dss = np.concatenate((dss_lh, dss_rh),axis=2)
    cnx_results = dense_connectivity_profile_isc(dss)
    np.save(os.path.join(results_dir, 'dense_connectivity_profile_isc.npy'), cnx_results)
    clf_results = searchlight_timepoint_clf(dss,window_size=5, buffer_size=10, NPROC=16)
    np.save(os.path.join(results_dir, 'time_segment_clf_accuracy.npy'), clf_results)

            


# perform leave-one-run-out cross validation on hyperalignment training
# this script 
if __name__ == '__main__':
    ha_type = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == 'budapest':
        import budapest_utils as utils
    elif dataset == 'raiders':
        import raiders_utils as utils
    elif dataset == 'whiplash':
        import whiplash_utils as utils
    else:
        print('dataset must be one of [whiplash,raiders,budapest]')
        sys.exit()
    print('running {a} on {b}'.format(a=ha_type,b=dataset))    
    all_runs = np.arange(1, utils.TOT_RUNS+1)
    
    # check if you specified which run you wanted to hold out.
    # otherwise, iterate through all train/test combos
    if len(sys.argv) > 3: 
        test = [int(sys.argv[3])]
        train = np.setdiff1d(all_runs, test)
        train_run_combos = [train]
    else:
        train_run_combos = list(itertools.combinations(all_runs, utils.TOT_RUNS-1))
 
    for train in train_run_combos: 
        test = np.setdiff1d(all_runs, train)
        print('training on runs {r}; testing on run {n}'.format(r=train, n=test))
        
        # separate testing and training data
        dss_train = utils.get_train_data('b',train)
        dss_test = utils.get_test_data('b', test)
        
        # get the node indices to run SL HA, both hemis
        node_indices = np.concatenate(prep.get_node_indices('b', surface_res=TOTAL_NODES))
        # get the surfaces for both hemis
        surface = prep.get_freesurfer_surfaces('b')
        # make the surface QE 
        qe = SurfaceQueryEngine(surface, HYPERALIGNMENT_RADIUS)
        
        # prepare the connectivity matrices and run HA if we are running CHA
        if ha_type == 'cha':
            target_indices = prep.get_node_indices('b', surface_res=SPARSE_NODES)
            dss_train = prep.compute_connectomes(dss_train, qe, target_indices)
            ha = SearchlightHyperalignment(queryengine=qe, 
                                           nproc=N_JOBS, 
                                           nblocks=N_BLOCKS, 
                                           mask_node_ids=node_indices, 
                                           dtype ='float64')
            Ts = ha(dss_train)
            outdir = os.path.join(utils.connhyper_dir, 'fold_{}/'.format(int(test[0])))
        
        # run response-based searchlight hyperalignment
        elif ha_type == 'rha':
            outdir = os.path.join(utils.resphyper_dir, 'fold_{}/'.format(int(test[0])))
            ha = SearchlightHyperalignment(queryengine=qe, 
                                           nproc=N_JOBS, 
                                           nblocks=N_BLOCKS, 
                                           mask_node_ids=node_indices, 
                                           dtype ='float64')
            Ts = ha(dss_train)
        
        # run hybrid hyperalignment
        elif ha_type == 'h2a':
            outdir = os.path.join(utils.h2a_dir, 'fold_{}/'.format(int(test[0])))
            ha = h2a.HybridHyperalignment(ref_ds=data[0], 
                             mask_node_indices=node_indices,
                             seed_indices=node_indices,
                             target_indices=target_indices,
                             target_radius=utils.HYPERALIGNMENT_RADIUS,
                             surface=surf)
            Ts = ha(dss_train)
        else:
            print('first argument must be one of h2a, cha, rha')
            sys.exit()

        save_transformations(Ts, os.path.join(outdir, 'transformations'))
        save_transformed_data(Ts, dss_test, os.path.join(outdir,'data') )
        run_benchmarks(ha_type, test[0], outdir)
        
       

