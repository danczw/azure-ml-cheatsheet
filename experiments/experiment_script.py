from azureml.core import Run
import argparse
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Get the experiment run context
run = Run.get_context() 	                                        # method to retrieve the experiment run context when the script is run

#-----EXPERIMENT_PARAMETER-----------------------------------------------------#
"""
* increase the flexibility of your training experiment by adding parameters to your script
* enabling you to repeat the same training experiment with different settings
"""
# Get the script arguments (regularization rate and training dataset ID)
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
# parser.add_argument('--input-data', type=str, dest='dataset_folder', help='data mount point') # Using file dataset
args = parser.parse_args()

# Set regularization hyperparameter (passed as an argument to the script)
reg = args.reg_rate

#-----DATA---------------------------------------------------------------------#
# load the dataset
print("Loading Data...")
diabetes = run.input_datasets['training_data'].to_pandas_dataframe()

# Using file dataset instead of tabular data:
    # data_path = run.input_datasets['training_files'] # Get the training data path from the input using a file dataset
    # (You could also just use args.dataset_folder if you don't want to rely on a hard-coded friendly name)
    # all_files = glob.glob(data_path + "/*.csv") # Read the files
    # diabetes = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#-----MODEL--------------------------------------------------------------------#
# Set regularization hyperparameter
reg = 0.01

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

#-----METRICS------------------------------------------------------------------#
# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', float(auc))

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

# Complete the run
run.complete()
