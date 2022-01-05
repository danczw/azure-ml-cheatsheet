from azureml.core import Workspace
from fairlearn.metrics import selection_rate, MetricFrame
from fairlearn.reductions import GridSearch, EqualizedOdds
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
data = ws.datasets.get('diabetes dataset')                      # Get specified dataset from list of all datasets in workspace
print(data)

#-----DATA_PREPROCESSING-------------------------------------------------------#
# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
X, y = data[features].values, data['Diabetic'].values

# Get sensitive features
S = data[['Age']].astype(int)
# Change value to represent age groups
S['Age'] = np.where(S.Age > 50, 'Over 50', '50 or younger')

# Split data into training set and test set
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.20, random_state=0, stratify=y)

#-----MODEL--------------------------------------------------------------------#
# Train a classification model
print("Training model...")
diabetes_model = DecisionTreeClassifier().fit(X_train, y_train)

print("Model trained.")

#-----UNFAIRNESS---------------------------------------------------------------#
'''
Unfairness
* Disparity between prediction rates or prediction performance metrics across sensitive feature groups
* Causes: data imbalance, indirect correlation, societal biases
* Fairlearn
    * Python package to analyze models and evaluate disparity between predictions and prediction performance for one or more sensitive features
	* Works by calculating group metrics for the sensitive features
	* Metrics themselves based on standard scikit-learn model evaluation metrics, such as accuracy, precision, or recall for classification models
    * Also provides support for mitigating unfairness in models
* Detect fairness:
    * Use fairlearn selection_rate function to return the selection rate (percentage of positive predictions) for the overall population
    * Use scikit-learn metric functions to calculate overall accuracy, recall, and precision metrics
    * Use a MetricFrame to calculate selection rate, accuracy, recall, and precision for each age group in the Age sensitive feature

Note: a mix of fairlearn and scikit-learn metric functions are used to calculate the performance values
'''
# Get predictions for the witheld test data
y_hat = diabetes_model.predict(X_test)

# Get overall metrics
print("Overall Metrics:")
# Get selection rate from fairlearn
overall_selection_rate = selection_rate(y_test, y_hat) # Get selection rate from fairlearn
print("\tSelection Rate:", overall_selection_rate)
# Get standard metrics from scikit-learn
overall_accuracy = accuracy_score(y_test, y_hat)
print("\tAccuracy:", overall_accuracy)
overall_recall = recall_score(y_test, y_hat)
print("\tRecall:", overall_recall)
overall_precision = precision_score(y_test, y_hat)
print("\tPrecision:", overall_precision)

# Get metrics by sensitive group from fairlearn
print('\nMetrics by Group:')
metrics = {'selection_rate': selection_rate,
           'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score}

group_metrics = MetricFrame(metrics=metrics,
                             y_true=y_test,
                             y_pred=y_hat,
                             sensitive_features=S_test['Age'])

print(group_metrics.by_group)

#-----UNFAIRNESS_MITIGATION----------------------------------------------------#
'''
Mitigating unfairness
* Balance training and validation data
    * Apply over-sampling or under-sampling techniques to balance data
    * Use stratified splitting algorithms to maintain representative proportions for training and validation
* Perform extensive feature selection and engineering analysis
    * Fully explore the interconnected correlations in data to try to differentiate features that are directly predictive from features that encapsulate more complex, nuanced relationships
	* Use the model interpretability support in Azure ML to understand how individual features influence predictions.
* Evaluate models for disparity based on significant features
    * Trade-off overall predictive performance for lower disparity in predictive performance between sensitive feature groups
    * --> 99.5% accuracy with comparable performance across all groups often more desirable than model that is 99.9% accurate but discriminates against a particular subset of cases

Fairlearn unfairnes mitigation algorithms
* Exponentiated Gradient
    * Reduction technique that applies a cost-minimization approach to learn optiamal trade.off of overall predictive performance and fairness disparity
    * Supports binary classification and regression
* Grid Search
    * Simplified version of Exponentiated Gradient algorithm
    * Works efficiently with small number of constraints
    * Supports binary classification and regression
* Threshold Optimizer
    * Post-provessing technique that applies contraint to existing classifier, transforming prediction as appropriate
    * Supports binary classification

Fairlearn Constrains
* Demographic parity
    * Use with any mitigation algorithms
    * Minimize disparity in selection rate across sensitive feature groups
    * Example binary classification scenario: ensure equal number of positive predictions in each group
* True positive rate parity
    * Use twith any mitigation algorithms
    * Minimize disparity in true positive rate across sensitive feature groups
    * Example binary classification scenario: ensure that each group contains comparable ratio of true positive predictions
* False-positive rate parity
    * Use with any mitigation algorithms
    * Minimize disparity in false_positive_rate across sensitive feature groups
    * Example binary classification scenario: ensure that each group contains a comparable ratio of false-positive predictions
* Equalized odds
    * Use with any of the mitigation algorithms
    * Minimize disparity in combined true positive rate and false_positive_rate across sensitive feature groups
    * Example binary classification scenario: ensure that each group contains a comparable ratio of true positive and false-positive predictions
* Error rate parity
    * Use with any reduction-based mitigation algorithms (Exponentiated Gradient and Grid Search)
    * Ensure that the error for each sensitive feature group does not deviate from the overall error rate by more than a specified amount
* Bounded group loss
    * Use with any of the reduction-based mitigation algorithms (Exponentiated Gradient and Grid Search)
    * Restrict the loss for each sensitive feature group in a regression model
'''
print('Finding mitigated models...')

# Train multiple models
sweep = GridSearch(
    DecisionTreeClassifier(),
    constraints=EqualizedOdds(),
    grid_size=20
)

sweep.fit(X_train, y_train, sensitive_features=S_train.Age)
models = sweep.predictors_

# Save the models and get predictions from them (plus the original unmitigated one for comparison)
model_dir = 'mitigated_models'
os.makedirs(model_dir, exist_ok=True)
model_name = 'diabetes_unmitigated'
print(model_name)
joblib.dump(value=diabetes_model, filename=os.path.join(model_dir, '{0}.pkl'.format(model_name)))
predictions = {model_name: diabetes_model.predict(X_test)}
i = 0
for model in models:
    i += 1
    model_name = 'diabetes_mitigated_{0}'.format(i)
    print(model_name)
    joblib.dump(value=model, filename=os.path.join(model_dir, '{0}.pkl'.format(model_name)))
    predictions[model_name] = model.predict(X_test)
