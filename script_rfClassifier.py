import argparse
import joblib
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# inference functions ---------------
def model_fn(model_dir):
	"""
	Function to do inference
	"""
	clf = joblib.load(os.path.join(model_dir, "model.joblib"))
	return clf


def _get_paramsRFClassifier():
	"""
	Function to get the hyperparameters founded in the notebook
	"""
	hypers = {'bootstrap': True,
			 'ccp_alpha': 0.0,
			 'class_weight': 'balanced_subsample',
			 'criterion': 'gini',
			 'max_depth': None,
			 'max_features': 0.5,
			 'max_leaf_nodes': None,
			 'max_samples': 498,
			 'min_impurity_decrease': 0.0,
			 'min_samples_leaf': 3,
			 'min_samples_split': 2,
			 'min_weight_fraction_leaf': 0.0,
			 'n_estimators': 60,
			 'n_jobs': -1,
			 'oob_score': True,
			 'random_state': None,
			 'verbose': 0,
			 'warm_start': False
			 }

	return hypers



def _train_RFClassifier(x,y,hyperparameters):
	"""
	Function to train the RF Classifier
	"""
	model = RandomForestClassifier(**hyperparameters)
	model.fit(x,y)
	return model


def _accuracy(model,x,y):
    """
    Function to compute accuracy
    """
    y_pred = model.predict(x)
    return (y_pred==y).mean()



if __name__ == "__main__":

	print("extracting arguments")
	parser = argparse.ArgumentParser()

	# Data, model, and output directories
	parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
	parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
	parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
	parser.add_argument("--train-file", type=str, default="trainValidation_clean.csv")
	parser.add_argument("--test-file", type=str, default="test_clean.csv")

	args, _ = parser.parse_known_args()

	# Getting rf´s hyperparamters
	hyperparameters = _get_paramsRFClassifier()

	args, _ = parser.parse_known_args()

	print("reading data")
	train_df = pd.read_csv(os.path.join(args.train, args.train_file))
	test_df = pd.read_csv(os.path.join(args.test, args.test_file))

	print("building training and testing datasets")
	x_train, y_train = train_df.drop(["Survived"],axis=1), train_df["Survived"] 

	test_size = 0.2
	random_state = 643
	x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, 
                                                  	  test_size = test_size,
                                                  	  random_state = random_state)

	x_test, y_test = test_df.drop(["Survived"],axis=1), test_df["Survived"] 

	# Train
	print("training model")
	model = _train_RFClassifier(x_train,y_train,hyperparameters)

	# Evaluation
	accT_RF_3 = _accuracy(model,x_train,y_train)
	accV_RF_3 = _accuracy(model,x_val,y_val)
	accTest_RF_3 = _accuracy(model,x_test,y_test)
	print("Random Forest 3 : Train ACCU : {:.3f}".format(accT_RF_3))
	print("Random Forest 3 : Valid ACCU : {:.3f}".format(accV_RF_3))
	print("Random Forest 3 : Test ACCU : {:.3f}".format(accTest_RF_3))

	# persist model
	path = os.path.join(args.model_dir, "model.joblib")
	joblib.dump(model, path)
	print("model persisted at " + path)

	# Test local
	# python script_rfClassifier.py --model-dir ./data/ --train ./data/ --test ./data/