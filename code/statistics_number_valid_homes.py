from test_homes import find_valid_homes

out = {}
for region in ['SanDiego','Austin']:
	out[region] ={}
	for appliance in ['fridge','hvac']:
		out[region][appliance] = {}
		for appliance_frac in [x/10.0 for x in range(2, 12, 2)]:
			out[region][appliance][appliance_frac] = {}
			for aggregate_frac in [x/10.0 for x in range(2, 12, 2)]:
				out[region][appliance][appliance_frac][aggregate_frac]=len(find_valid_homes(region, appliance, appliance_frac, aggregate_frac))
