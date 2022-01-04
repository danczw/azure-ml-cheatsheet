# Import libraries
from azureml.core import Run, Model
import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get the experiment run context
run = Run.get_context() 	                                        # method to retrieve the experiment run context when the script is run

#-----EXPERIMENT_PARAMETER-----------------------------------------------------#
'''
Experiment Parameter
* Increase the flexibility of your training experiment by adding parameters to your script
* Enabling to repeat the same training experiment with different settings
'''
parser = argparse.ArgumentParser()
parser.add_argument('--training-data', type=str, dest='training_data', help='training data')
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
args = parser.parse_args()                                          # Add arguments to args collection

# Set training data from prepared data
training_data = args.training_data

# Set regularization hyperparameter (passed as an argument to the script)
reg = args.reg_rate

#-----DATA---------------------------------------------------------------------#
# load the prepared data file in the training folder
print('Loading Data...')
file_path = os.path.join(training_data,'data.csv')
diabetes = pd.read_csv(file_path)

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#-----MODEL--------------------------------------------------------------------#
# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  float(reg))
model = LogisticRegression(C=1/reg, solver='liblinear').fit(X_train, y_train)

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

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
run.log_image(name = 'ROC', plot = fig)

#-----SAVE_MODEL---------------------------------------------------------------#
# Save the trained model in the outputs folder
print('Saving model...')
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'diabetes_model.pkl')
joblib.dump(value=model, filename=model_file)

# Register the model
print('Registering model...')
Model.register(
    workspace=run.experiment.workspace,
    model_path = model_file,
    model_name = 'diabetes_model',
    tags={'Training context':'Pipeline'},
    properties={
        'AUC': float(auc),
        'Accuracy': float(acc)
    }
)

run.complete()