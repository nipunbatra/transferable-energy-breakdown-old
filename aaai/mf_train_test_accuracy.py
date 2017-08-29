from sklearn.model_selection import LeaveOneOut

from aaai18.common import compute_rmse
from create_matrix import *
from mf_core import *

APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014


pred = {}
test_error = {}
for appliance in APPLIANCES[1:2]:
	if appliance == "hvac":
		start, stop = 5, 11
	else:
		start, stop = 1, 13
	pred[appliance] = {}
	test_error[appliance]={}
	appliance_df = create_matrix_region_appliance_year(region, year, appliance)
	static_cols = ['area', 'total_occupants', 'num_rooms']
	aggregate_cols = [x for x in appliance_df.columns if "aggregate" in x]
	appliance_cols = [x for x in appliance_df.columns if appliance in x]
	energy_cols = np.concatenate([aggregate_cols, appliance_cols])

	df = appliance_df.copy()
	dfc = df.copy()

	df = df[energy_cols]
	col_max = df.max().max()
	col_min = df.min().min()
	df = (1.0*(df-col_min))/(col_max-col_min)
	X_cols = np.concatenate([aggregate_cols, static_cols])

	for features in ['energy', 'energy_static'][:1]:
		pred[appliance][features] = {}
		test_error[appliance][features]={}
		if features == "energy":
			cols = aggregate_cols
		else:
			cols = X_cols

		X_matrix, X_normalised, matrix_max, matrix_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance,
		                                                                                            col_max,
		                                                                                            col_min, False)
		static_features = get_static_features(dfc, X_normalised)
		if features=="energy":
			feature_comb = ['None']
		else:
			feature_comb =['occ','area','rooms']
		idx_user, data_user = prepare_known_features(feature_comb, static_features, X_normalised)

		for cost in ['absolute','relative'][:]:
		#for cost in ['absolute']:
			pred[appliance][features][cost] = {}
			test_error[appliance][features][cost] ={}
			for latent_factors in range(3, 7):
				pred[appliance][features][cost][latent_factors] = {}
				test_error[appliance][features][cost][latent_factors] ={}

				print latent_factors, features, appliance, cost
				loo = LeaveOneOut()
				for train_ix, test_ix in loo.split(appliance_df):
					test_home = appliance_df.index.values[test_ix][0]
					pred[appliance][features][cost][latent_factors][test_home] = {}
					A = create_matrix_factorised(appliance, [test_home], X_normalised)
					X, Y, res = nmf_features(A=A, k=latent_factors, constant=0.01, regularisation=False,
					                         idx_user=idx_user, data_user=data_user,
					                         idx_item=None, data_item=None, MAX_ITERS=10, cost=cost)
					p = pd.DataFrame(Y * X) * col_max+col_min
					p.index = X_normalised.index
					p.columns = X_normalised.columns
					test_error[appliance][features][cost][latent_factors][test_home]=compute_rmse(appliance, p[appliance_cols])
					pred_df = create_prediction(test_home, X, Y, X_normalised, appliance,
					                            col_max, col_min, appliance_cols)
					pred[appliance][features][cost][latent_factors][test_home] = pred_df

				pred[appliance][features][cost][latent_factors] = pd.DataFrame(pred[appliance][features][cost][latent_factors]).T
				test_error[appliance][features][cost][latent_factors] = pd.Series(test_error[appliance][features][cost][latent_factors])
			test_error[appliance][features][cost] = pd.DataFrame(test_error[appliance][features][cost]).mean()
#pickle.dump(pred, open('predictions/mf.pkl', 'w'))
