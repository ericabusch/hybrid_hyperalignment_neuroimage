"""
HA_prep_functions.py
An amalgamation of functions that were useful to set up hyperalignment for the H2A analyses.

see :ref: `Busch, Slipski, et al., NeuroImage (2021)` for details.

February 2021
@author: Erica L. Busch
"""
import os
import numpy as np
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.fxy import FxyMapper
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.measures.searchlight import Searchlight
from mvpa2.mappers.zscore import zscore
import scipy.stats

MASKS = {'l': np.load(os.path.join(basedir, 'fsaverage_lh_mask.npy')), 'r': np.load(os.path.join(basedir, 'fsaverage_rh_mask.npy'))}


class MeanFeatureMeasure(Measure):
    """Mean group feature measure
    Because the vanilla one doesn't want to work for Swaroop and I adapted this from Swaroop.
    Debugging is hard. Accepting the struggles of someone smarter than me is easy.
    """
    is_trained = True

    def __init__(self, **kwargs):
        Measure.__init__(self, **kwargs)

    def _call(self, dataset):
        return Dataset(samples=np.mean(dataset.samples, axis=1))

    
    
def compute_seed_means(measure, queryengine, ds, roi_ids):
    # Seed data is the mean timeseries for each searchlight
    seed_data = Searchlight(measure, queryengine=queryengine, 
                            nproc=1, roi_ids=roi_ids.copy())
    if isinstance(ds,np.ndarray):
        ds = Dataset(ds) 
    seed_data = seed_data(ds)
    zscore(seed_data.samples, chunks_attr=None)
    return seed_data

def compute_connectomes(datasets, queryengine, target_indices):
    conn_metric = lambda x,y: np.dot(x.samples, y.samples)/x.nsamples
    connectivity_mapper = FxyMapper(conn_metric)
    mean_feature_measure = MeanFeatureMeasure()

    # compute means for aligning seed features
    conn_means = [seed_means(MeanFeatureMeasure(), queryengine, ds, target_indices) for ds in datasets]

    conn_targets = []
    for csm in conn_means:
        zscore(csm, chunks_attr=None)
        conn_targets.append(csm)

    connectomes = []
    for target, ds in zip(conn_targets, datasets):
        conn_mapper.train(target)
        connectome = connectivity_mapper.forward(ds)
        connectome.fa = ds.fa
        zscore(connectome, chunks_attr=None)
        connectomes.append(connectome)
    return connectomes

# determines which node indices belong to a surface of a given resolution,
# excluding the medial wall.
def get_node_indices(hemi, surface_res=None):
    if surface_res == None:
        surface_res = utils.SURFACE_RESOLUTION
    if hemi == 'b':
        r = get_node_indices('r', surface_res=surface_res)
        l = get_node_indices('l', surface_res=surface_res)
        r = r + TOT_NODES
        return [l,r]
    mask = MASKS[hemi]
    idx = np.where(mask[:surface_res])[0]
    return idx

# uses .white and .pial files to create a Surface
def get_freesurfer_surfaces(hemi):
    import nibabel as nib
    from mvpa2.support.nibabel.surf import Surface
    if hemi == 'b':
        lh = get_freesurfer_surfaces('l')
        rh = get_freesurfer_surfaces('r')
        return lh.merge(rh)
    coords1, faces1 = nib.freesurfer.read_geometry(os.path.join(utils.basedir,'{lr}h.white'.format(lr=hemi)))
    coords2, faces2 = nib.freesurfer.read_geometry(os.path.join(utils.basedir,'{lr}h.pial'.format(lr=hemi)))
    np.testing.assert_array_equal(faces1, faces2)
    surf = Surface((coords1 + coords2) * 0.5, faces1)
    return surf


# This function uses pymvpa surface query engines to create a list of lists, where each sub-list is all 
# the nodes within a searchlight of radius X centered on the first node in the list.
def get_searchlights(hemi,radius):
    if radius is None:
        radius = utils.SEARCHLIGHT_RADIUS 
    savename = os.path.join(utils.basedir,'{R}mm_searchlights_{S}h.npy'.format(R=radius,S=hemi))
    try:
        return np.load(savename)
    from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
    # get the data for jsut the first participant
    node_indices = get_node_indices(hemi)
    surf = get_freesurfer_surfaces(hemi)
    subj = utils.subjects[0]
    # get one run of one subject
    ds = get_train_data(hemi, 1, num_subjects=1)[0]
    ds.fa['node_indices'] = node_indices.copy()
    qe = SurfaceQueryEngine(surf, radius)
    qe.train(ds)

    searchlights = []
    for idx in node_indices:
        sl = qe.query_byid(idx)
        searchlights.append(sl)
   
    np.save(savename, searchlights)
    return searchlights


