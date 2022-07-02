# Import libraries
from azureml.core.run import Run
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Method to retrieve the experiment run context when the script is run
run = Run.get_context()

#-----DATA---------------------------------------------------------------------#
# load the data (passed as an input dataset)
print("Loading Data...")
data = run.input_datasets['training_data'].to_pandas_dataframe()

# Separate features and labels
features = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
labels = ['not-diabetic', 'diabetic']
X, y = data[features].values, data['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30,
                                                    random_state=0
                                                   )

#-----MODEL--------------------------------------------------------------------#
# Train a decision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

#-----METRICS------------------------------------------------------------------#
# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
run.log('AUC', np.float(auc))

#-----SAVE_MODEL---------------------------------------------------------------#
# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/diabetes.pkl')

#-----EXPLANATION--------------------------------------------------------------#
'''
Feature Importance 
* Creating explainers - there are multiple types of explainer, including:
	* MimicExplainer:
        * Creates a global surrogate model that approximates your trained model and can be used to generate explanations
        * Explainable model must have the same kind of architecture as your trained model (for example, linear or tree-based)
    * TabularExplainer:
        * Acts as wrapper around various SHAP explainer algorithms
        * Automatically choosing the one that is most appropriate for model architecture
    * PFIExplainer:
        * Permutation Feature Importance explainer
        * Analyzing feature importance by shuffling feature values and measuring the impact on prediction performance.
* To retrieve global importance values for the features call the explain_global() method of your explainer to get a global explanation
* Use get_feature_importance_dict() method to get a dictionary of the feature importance values

Feature Importance
* Global feature importance
    * Quantifies the relative importance of each feature in the test dataset as a whole
	* Provides a general comparison of the extent to which each feature in the dataset influences prediction
* Local feature importance
    * Measures influence of each feature value for a specific individual prediction
	* Multi-class classification model:
        * Local importance values for each possible class is calculated for every feature
        * Total importance value across all classes of this prediction always being 0
	* Regression model:
        * Local importance values simply indicate the level of influence each feature has on the predicted scalar label
'''
# Get explanation
explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

# Complete the run
run.complete()