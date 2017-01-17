from test_homes import find_valid_homes
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt

out = {}
for region in ['SanDiego','Austin']:
	out[region] ={}
	for appliance in ['fridge','hvac']:
		out[region][appliance] = {}
		for appliance_frac in [x/10.0 for x in range(2, 12, 2)]:
			out[region][appliance][appliance_frac] = {}
			for aggregate_frac in [x/10.0 for x in range(2, 12, 2)]:
				out[region][appliance][appliance_frac][aggregate_frac]=len(find_valid_homes(region, appliance, appliance_frac, aggregate_frac))


out_df_sd_fridge = pd.DataFrame(out['SanDiego']['fridge'])
out_df_sd_fridge.index.name = "Aggregate Fraction"

for region in ['SanDiego','Austin']:
	for appliance in ['fridge','hvac']:

		sns.heatmap(pd.DataFrame(out[region][appliance]), annot=True)
		plt.savefig("../results/"+region+appliance+".png")
		plt.clf()

