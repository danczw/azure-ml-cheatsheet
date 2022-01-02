# Import libraries
import argparse, joblib, os
from azureml.core import Run
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Get the experiment run context
run = Run.get_context()

#-----EXPERIMENT_PARAMETER-----------------------------------------------------#
'''
* increase the flexibility of your training experiment by adding parameters to your script
* enabling you to repeat the same training experiment with different settings
'''
parser = argparse.ArgumentParser()

# Input dataset
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

# Hyperparameters
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='learning rate')
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')
args = parser.parse_args()                                      # Add arguments to args collection

#-----DATA---------------------------------------------------------------------#
# load the data (passed as an input dataset)
print("Loading Data...")
diabetes = run.input_datasets['training_data'].to_pandas_dataframe()

#-----DATA_PREP----------------------------------------------------------------#
# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#-----HYPERPARAMETER-----------------------------------------------------------#
# Log Hyperparameter values
run.log('learning_rate',  np.float(args.learning_rate))
run.log('n_estimators',  np.int(args.n_estimators))

#-----METRICS------------------------------------------------------------------#
# Train a Gradient Boosting classification model with the specified hyperparameters
print('Training a classification model')
model = GradientBoostingClassifier(learning_rate=args.learning_rate,
                                   n_estimators=args.n_estimators).fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

#-----SAVE_MODEL---------------------------------------------------------------#
# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()