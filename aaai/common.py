from sklearn.metrics import mean_squared_error
from create_matrix import *



def compute_rmse(appliance, pred_df, region='Austin',year=2014):
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)

	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred_df = pred_df.copy()
	pred_df.columns = [['%s_%d' % (appliance, month) for month in range(start, stop)]]
	gt_df = appliance_df[pred_df.columns].ix[pred_df.index]



	gt_df = gt_df.unstack().dropna()
	pred_df = pred_df.unstack().dropna()
	pred_df = pred_df.ix[gt_df.index]

	rms = np.sqrt(mean_squared_error(gt_df, pred_df))
	return rms


def compute_rmse_fraction(appliance, pred_df, region='Austin',year=2014):
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)

	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred_df = pred_df.copy()
	pred_df.columns = [['%s_%d' % (appliance, month) for month in range(start, stop)]]
	gt_df = appliance_df[pred_df.columns].ix[pred_df.index]

	aggregate_df = appliance_df.ix[pred_df.index][['aggregate_%d' % month for month in range(start, stop)]]

	aggregate_df.columns = gt_df.columns
	rows, cols = np.where((aggregate_df < 100))
	for r, c in zip(rows, cols):
		r_i, c_i = aggregate_df.index[r], aggregate_df.columns[c]
		aggregate_df.loc[r_i, c_i] = np.NaN

	gt_fraction = gt_df.div(aggregate_df) * 100
	pred_fraction = pred_df.div(aggregate_df) * 100

	gt_fraction_dropna = gt_fraction.unstack().dropna()
	pred_fraction_dropna = pred_fraction.unstack().dropna()
	pred_fraction_dropna = pred_fraction_dropna.ix[gt_fraction_dropna.index]
	difference_error = (gt_fraction_dropna-pred_fraction_dropna).abs()

	rms = np.sqrt(mean_squared_error(gt_fraction_dropna, pred_fraction_dropna))
	return gt_fraction, pred_fraction, rms, difference_error

import os

def get_directory_structure(rootdir):
	"""
	Creates a nested dictionary that represents the folder structure of rootdir
	"""
	dir = {}
	rootdir = rootdir.rstrip(os.sep)
	start = rootdir.rfind(os.sep) + 1
	for path, dirs, files in os.walk(rootdir):
		folders = path[start:].split(os.sep)
		if len(files):
			subdir = pd.read_csv(files[0], index_col=0)
		else:
			subdir = dict.fromkeys(files)
		parent = reduce(dict.get, folders[:-1], dir)
		parent[folders[-1]] = subdir
	return dir