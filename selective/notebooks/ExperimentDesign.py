from sklearn.model_selection import train_test_split, KFold
import sys
from tensor_custom_core import *
sys.path.insert(0, '../../aaai18/code/')
from common import *
from create_matrix import *
import random

# initialization
appliance_index = {appliance: APPLIANCES_ORDER.index(appliance) for appliance in APPLIANCES_ORDER}
APPLIANCES = ['fridge', 'hvac', 'wm', 'mw', 'oven', 'dw']
region = "Austin"
year = 2014
case = 2
num_home = 3
num_season = 3
source = 'Austin'
target = 'SanDiego'
constant_use = 'True'
start = 1
stop = 13
train_percentage = 50
validation_percentage = 10
test_percentage = 40

source_df, source_dfc, source_tensor, source_static = create_region_df_dfc_static(source, year, start, stop)
target_df, target_dfc, target_tensor, target_static = create_region_df_dfc_static(target, year, start, stop)

num_samples = len(source_df)
index_list = np.arange(num_samples)
train_loc, validate_test_loc = train_test_split(index_list, train_size=train_percentage/100, random_state=0)
validate_loc, test_loc = train_test_split(validate_test_loc, train_size=2*validation_percentage/100, random_state=0)
print test_loc

train_df = source_df.loc[train_loc]
validate_df = source_df.loc[validate_loc]
test_df = source_df.loc[test_loc]

print validate_df