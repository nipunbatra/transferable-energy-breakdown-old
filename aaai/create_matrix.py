import pandas as pd
import numpy as np
import sys
sys.path.append("../code/")
from common_functions import create_region_df

upper_limit = {
	'hvac': 40.,
	'fridge': 10.,
	'wm': 1.,
	'oven':1.,
	'mw':1.,
	'dw':1.

}

def create_matrix_single_region(region, year):
	temp_df, temp_dfc = create_region_df(region, year)
	return temp_df, temp_dfc


def create_matrix_all_entries(region, year, appliance):
	df, dfc = create_matrix_single_region(region, year)
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	appliance_cols = ['%s_%d' % (appliance, month) for month in range(start, stop)]
	aggregate_cols = ['%s_%d' % ("aggregate", month) for month in range(start, stop)]
	static_cols = ['area', 'total_occupants', 'num_rooms']
	all_cols = np.concatenate(np.array([appliance_cols, aggregate_cols, static_cols]).flatten())
	df = df[all_cols].dropna()
	df = df[df["total_occupants"] > 0]
	df = df[df["area"] > 100]
	df = df[~(df[aggregate_cols] < 200).sum(axis=1).astype('bool')]
	ul_appliance = upper_limit[appliance]
	df = df[~(df[appliance_cols] < ul_appliance).sum(axis=1).astype('bool')]

	return df

