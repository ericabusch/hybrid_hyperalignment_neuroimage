""" hybrid hyperalignment on surface.

see :ref: `Busch, Slipski, et al., NeuroImage (2021)` for details.

February 2021
@author: Erica L. Busch

"""
import os
import numpy as np
import scipy.stats
from mvpa2.mappers.zscore import zscore

from mvpa2.base import debug
from mvpa2.measures.base import Measure
from mvpa2.base.hdf5 import h5save, h5load

from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.measures.searchlight import Searchlight
from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.fxy import FxyMapper

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

    
    
class HybridHyperalignment():
    """
    Given a list of datasets, provide a list of mappers into common space based on hybrid hyperalignment.
    
    1) Input datasets should be PyMVPA datasets.
    2) They should be of the same size (nsamples, nfeatures) 
    and be aligned anatomically. 
    3) All features in the datasets should be zscored.
    4) Datasets should all have a feature attribute `node_indices` containing the location of the feature
    on the surface. 
    
    Parameters
    ----------
    mask_ids: Default is none, type is list or array of integers. Specify a mask within which to compute 
    searchlight hyperalignment. If none, set equal to seed_indices. One of the two is required.
    
    seed_indices: Default is none, type is list or array of integers. Node indices that correspond to seed 
    centers for connectivity seeds. If none, set equal to mask_ids. One of the two is required.
    
    target_indices: Default is none, type is list or array of integers. Node indices corresponding to the center
    of connectivity targets. If none, set equal to seed_indices (will be dense!).
    
    surface: Required. The freesurfer surface defining your data.
    
    queryengine: Required. A single pymvpa query engine (or list of pymvpa queryengines, one per dataset) to be used by 
    searchlight hyperalignment. If none, will be defined from surface and seed radius. 
    
    target_radius: default is 20mm ala H2A paper. Minimum is 1. Radius for target searchlight.
    
    seed_radius: default is 13 ala H2A paper. Radius for connectivity seed searchlight.
    
    conn_metric: Connectivity metric between features. Default is the dot product of samples (which on zscored data 
    becomes correlation if you normalize by nsamples.
    
    dtype: default is 'float64'.
    
    nproc: default is 1. 
    
    nblocks: Number of blocks to divide to process. Higher number means lower memory consumption. default is 1. 
    
    get_all_mappers: do you want to return both the mappers from AA -> iter1 space, and iter1_space -> H2A space? 
    defualts to false.
    
    Returns
    -------
    if get_all_mappers, returns iteration1 mappers, iteration2 mappers, and the final mappers in a list for each subject. 
    otherwise, returns only the final mappers.
    
    """
    
  
    
    def __init__(self, ref_ds, surface, mask_node_indices=None, seed_indices=None, target_indices=None, queryengine=None, target_radius=20, seed_radius=13, dtype='float64', nproc=1, nblocks=1, get_all_mappers=False):        
        self.ref_ds = ref_ds 
        self.mask_node_indices = mask_node_indices
        self.seed_indices = seed_indices
        self.target_indices = target_indices
        self.surface = surface 
        self.queryengine = queryengine 
        self.target_radius = target_radius 
        self.seed_radius = seed_radius 
        self.conn_metric = lambda x, y: np.dot(x.samples.T, y.samples)/x.nsamples
        self.dtype = dtype 
        self.nproc = nproc #"""Number of blocks to divide to process. Higher number means lower memory consumption."""
        self.nblocks = nblocks  #"""Number of blocks to divide to process. Higher number means lower memory consumption."""
        self.target_queryengine = None
        self.get_all_mappers = get_all_mappers 
        
        if self.seed_indices is None:
            self.seed_indices = np.arange(ref_ds.shape[-1])
            
        if self.target_indices is None:
            self.target_indices = np.arange(ref_ds.shape[-1])
        
        if self.mask_node_indices is None:
            self.mask_node_indices = self.seed_indices.copy()
        
        if self.queryengine is None:
            self.queryengine = SurfaceQueryEngine(self.surface, self.seed_radius)
            self.queryengine.train(ref_ds)
        
        if self.target_queryengine is None:
            self.target_queryengine = SurfaceQueryEngine(self.surface, self.target_radius)
            self.target_queryengine.train(ref_ds)
      
    def _seed_means(self, measure, queryengine, ds, seed_indices):
        # Seed data is the mean timeseries for each searchlight
        seed_data = Searchlight(measure, queryengine=queryengine, 
                                nproc=self.nproc, roi_ids=np.concatenate(seed_indices).copy())
        if isinstance(ds,np.ndarray):
            ds = Dataset(ds) 
        seed_data = seed_data(ds)
        zscore(seed_data.samples, chunks_attr=None)
        return seed_data
    
    def _get_connectomes(self, datasets):
        conn_mapper = FxyMapper(self.conn_metric)
        mean_feature_measure = MeanFeatureMeasure()
        qe = self.queryengine
        
        roi_ids = self.target_indices
        # compute means for aligning seed features
        conn_means = [self._seed_means(MeanFeatureMeasure(), qe, ds, roi_ids) for ds in datasets]
        
        conn_targets = []
        for csm in conn_means:
            zscore(csm, chunks_attr=None)
            conn_targets.append(csm)
        
        connectomes = []
        for target, ds in zip(conn_targets, datasets):
            conn_mapper.train(target)
            connectome = conn_mapper.forward(ds)
            connectome.fa = ds.fa
            zscore(connectome, chunks_attr=None)
            connectomes.append(connectome)
        return connectomes
    
    def _apply_mappers(self, datasets, mappers):
        aligned_datasets = [d.get_mapped(M) for d,M in zip(datasets, mappers)]
        return aligned_datasets
    
    def _frobenius_norm_and_merge(self, dss_connectomes, dss_response, node_indices):
        # figure out which of the two types of data are larger
        if  dss_response[0].shape[0] > dss_connectomes[0].shape[0]:
            larger = dss_response
            smaller = dss_connectomes
        else:
            larger = dss_connectomes
            smaller = dss_response
        node_ids = node_indices
        # find the normalization ratio based on which is larger 
        norm_ratios = []
        for la, sm in zip(larger, smaller):
            laN = np.linalg.norm(la, ord='fro')
            smN = np.linalg.norm(sm, ord='fro')
            v = laN / smN
            norm_ratios.append(v)

        # normalize the smaller one and then merge the datasets
        merged_dss = []
        for la, sm, norm in zip(larger, smaller, norm_ratios):
            d_sm = sm.samples * norm
            merged = np.vstack((d_sm, la.samples))
            merged = Dataset(samples=merged)
            merged.fa['node_indices'] = node_ids.copy()
            merged_dss.append(merged)
        return merged_dss
    
    def _prep_h2a_data(self, response_data, node_indices):
        for d in response_data:
            if isinstance(d, np.ndarray):
                d = Dataset(d)
            d.fa['node_indices']= node_indices.copy()
        
        connectivity_data = self._get_connectomes(response_data)
        h2a_input_data = self._frobenius_norm_and_merge(connectivity_data, response_data, node_indices)
        for d in h2a_input_data:
            d.fa['node_indices'] = node_indices.copy()
            zscore(d, chunks_attr=None)
        return h2a_input_data
    
    def __call__(self, datasets):
        """ estimate mappers for each dataset.
        Parameters
        ---------- 
        datasets : list of datasets
        
        Returns 
        -------
        mappers_iter1: mappers from the first HA iteration
        mappers_iter2: mappers from the second HA iteration
        
        """
        debug.active += ['SHPAL', 'SLC']

        mask_node_indices = np.concatenate(self.mask_node_indices)
        qe = self.queryengine
        nproc = self.nproc
        dtype = self.dtype
        nblocks = self.nblocks
                
        ha_iter1 = SearchlightHyperalignment(queryengine=qe, 
                                      nproc=nproc, 
                                      nblocks=nblocks, 
                                      mask_node_ids=mask_node_indices,
                                      dtype=dtype)
        
        mappers_iter1 = ha_iter1(datasets)
        aligned_iter1_datasets = self._apply_mappers(datasets, mappers_iter1)
        
        
        h2a_input_data = self._prep_h2a_data(aligned_iter1_datasets, mask_node_indices)
        ha_iter2 = SearchlightHyperalignment(queryengine=qe, 
                                      nproc=nproc, 
                                      nblocks=nblocks, 
                                      mask_node_ids=mask_node_indices,
                                      dtype=dtype)
        mappers_iter2 = ha_iter2(h2a_input_data)
        # push the original data through the trained model
        mappers_final = ha_iter2(datasets)
        
        if self.get_all_mappers:
            return mappers_iter1, mappers_iter2, mappers_final
        return mappers_final
        
        
        
        
            
    
     
    
    `
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    