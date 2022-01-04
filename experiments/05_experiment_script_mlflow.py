import argparse
from azureml.core import Run
import mlflow
import pandas as pd

# Get the experiment run context
run = Run.get_context() 	                                        # method to retrieve the experiment run context when the script is run

#-----EXPERIMENT_PARAMETER-----------------------------------------------------#
'''
Experiment parameter
* increase the flexibility of your training experiment by adding parameters to your script
* enabling you to repeat the same training experiment with different settings
'''
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-data',
    type=str,
    dest='raw_dataset_id',
    help='raw dataset'
)

args = parser.parse_args()                                      # Add arguments to args collection

#-----DATA---------------------------------------------------------------------#
# load the data (passed as an input dataset)
print('Loading Data...')
data = run.input_datasets['raw_data'].to_pandas_dataframe()

#-----MLFLOW_EXPERIMENT--------------------------------------------------------#
'''
MLflow
* Open-source product designed to manage the Machine Learning development lifecycle
* MLflow Tracking -  log parameters, versions of libraries used, evaluation metrics, generated output files, etc.
    * Parameter: Key-value pairs representing represent inputs (hyperparameter, function vars, etc.)
    * Metrics: Key-value pairs representing how the model is performing
    * Artifacts: Output files
* MLflow Projects -  packaging up code in a manner, which allows for consistent deployment and the ability to reproduce results
* MLflow Models - standardized format for packaging models for distribution
* MLflow Model Registry - register models in a registry
'''
# start the MLflow experiment
with mlflow.start_run():

    # Count the rows and log the result
    row_count = (len(data))
    print('observations:', row_count)
    mlflow.log_metric('observations', row_count)

    # Count and log the label counts
    diabetic_counts = data['Diabetic'].value_counts()
    print(diabetic_counts)
    for k, v in diabetic_counts.items():
        mlflow.log_metric('Label:' + str(k), v)