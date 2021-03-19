#apply tests of fit to anatomically aligned data as control
#EB 7.2020

import sys, os, glob
import numpy as np
import benchmarks 


if __name__ == '__main__':
	dataset = sys.argv[1]
	if dataset == 'budapest':
		import budapest_utils as utils
	elif dataset == 'whiplash':
		import whiplash_utils as utils
	elif dataset == 'raiders':
		import raiders_utils as utils
	else:
		print('dataset must be one of [raiders, whiplash, budapest]')
		sys.exit(2)
	test_run = int(sys.argv[2])
	outdir = os.path.join(utils.basedir,'anatomical',dataset,'fold_{x}'.format(x=run))
	if not os.path.isdir(outdir):
		os.makedirs(outdir)
	test_lh = utils.get_test_data('lh',[run])
	test_rh = utils.get_test_data('rh',[run])
	lh_res, rh_res = benchmarks.vertex_isc(test_lh), benchmarks.vertex_isc(test_rh)
	np.save(outdir+'/vertex_isc_lh.npy', lh_res)
	np.save(outdir+'/vertex_isc_rh.npy', rh_res)
	test_data = np.concatenate((test_lh,test_rh),axis=2)
	res = benchmarks.dense_connectivity_profile_isc(test_data)
	np.save(outdir+'/dense_connectome_isc.npy', res)
	res = benchmarks.searchlight_timepoint_clf(test_data)


	