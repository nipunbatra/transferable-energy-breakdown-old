import numpy as np
import pandas as pd
import tensorflow as tf

from tffm import TFFMRegressor
from tffm import utils
from collections import OrderedDict, defaultdict
from sklearn import metrics, preprocessing

RANDOM_STATE = 0


def train_examples_from_df(frame, metadata):
	one_hot_encoder = {}

	# Create and fit one hot encoder for column labels
	lb = preprocessing.LabelBinarizer()
	lb.fit(frame.columns)
	one_hot_encoder['Column'] = lb

	# Create and fit one hot encoder for row labels
	lb = preprocessing.LabelBinarizer()
	lb.fit(frame.index)
	one_hot_encoder['Row'] = lb

	X = defaultdict(list)

	for row in frame.index:

		# Initialize features with row metadata
		row_metadata = [item for item in metadata.loc[row]]
		features = row_metadata

		# Add one hot encoded row label
		features.extend(one_hot_encoder['Row'].transform([row])[0])

		for col in frame.columns:

			# If the entry at current position is NaN, we can't use it
			if np.isnan(frame.loc[row, col]): continue

			train_example = features[:]

			# Add one hot encoded column label
			train_example.extend(one_hot_encoder['Column'].transform([col])[0])

			# Add value to be predicted
			train_example.append(frame.loc[row, col])

			# Insert the train example and column label in dict indexed by row labels
			X[row].append((train_example, col))

	return X


def factorization_machine(full_frame, metadata_frame, maximum, minimum):
	X = train_examples_from_df(full_frame, metadata_frame)

	predicted_df = pd.DataFrame(index=full_frame.index, columns=full_frame.columns)
	ground_truth_df = pd.DataFrame(index=full_frame.index, columns=full_frame.columns)

	for row in full_frame.index:

		# All rows except current row form the training set
		X_train = np.array([example[:-1] for k in X for example, col in X[k] if k != row])
		y_train = np.array([example[-1] for k in X for example, col in X[k] if k != row])

		# Current row forms the test set
		X_test = np.array([example[:-1] for example, col in X[row]])
		y_test = np.array([example[-1] for example, col in X[row]])

		# y_cols will contain the Column labels
		# This will be used to reconstruct the predicted and gt df
		y_cols = [col for example, col in X[row]]

		fmachine = TFFMRegressor(
			order=2,
			rank=64,
			optimizer=tf.train.AdagradOptimizer(learning_rate=0.1),
			n_epochs=100,
			batch_size=-1,
			init_std=0.01,
			input_type='dense',
			reg=4,
			seed=RANDOM_STATE,
			verbose=0
		)

		fmachine.fit(X_train, y_train)

		y_pred = fmachine.predict(X_test)

		# Insert predicted and gt values into the df
		for idx, col in enumerate(y_cols):
			predicted_df.loc[row, col] = y_pred[idx]
			ground_truth_df.loc[row, col] = y_test[idx]

		predicted_df.loc[row] = predicted_df.loc[row] * (maximum - minimum) + minimum
		print predicted_df.loc[row], row
	return predicted_df
