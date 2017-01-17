from common_functions import  create_region_df
from copy import deepcopy
import  numpy as np

REGIONS = ['SanDiego','Austin']
APPLIANCES = ['hvac','fridge']

valid_homes_data = {}


def find_valid_homes(region, appliance, appliance_fraction, aggregate_fraction):
	df, dfc = create_region_df(region)
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	appliance_cols = ['%s_%d' % (appliance, month) for month in range(start, stop)]
	aggregate_cols = ['aggregate_%d' % month for month in range(start, stop)]

	aggregate_num_cols, appliance_num_cols = len(aggregate_cols), len(appliance_cols)

	# Choose homes that have 1) atleast half the aggregate_cols and
	# 2) half the appliance cols

	# Condition #1
	valid_homes_aggregate = df.ix[df[aggregate_cols].notnull().sum(axis=1) >= aggregate_num_cols*aggregate_fraction].index

	# Condition #2
	valid_homes_appliance = df.ix[df[appliance_cols].notnull().sum(axis=1) >= appliance_num_cols*appliance_fraction].index

	# Homes meeting both
	valid_homes = np.intersect1d(valid_homes_aggregate, valid_homes_appliance)
	return valid_homes


valid_homes_data = {}
for region in REGIONS:
	valid_homes_data[region] = {}
	for appliance in APPLIANCES:
		valid_homes_data[region][appliance] = find_valid_homes(region, appliance, 1.0, 1.0)


"""
for region in ['SanDiego']:
	valid_homes_data[region] = {}
	df, dfc = create_region_df(region)
	for appliance in APPLIANCES:
		if appliance=="hvac":
			start, stop = 5, 11
		else:
			start, stop=1, 13


		valid_homes_data[region][appliance] = valid_homes
"""
