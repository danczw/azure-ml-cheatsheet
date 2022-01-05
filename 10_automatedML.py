# Import libraries
from azureml.core import Environment, Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
import azureml.train.automl.utilities as automl_utils
from azureml.widgets import RunDetails

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')               # Get specified dataset from list of all datasets in workspace

# Split for train and test dataset
train_ds, test_ds = diabetes_ds.random_split(percentage=0.7, seed=123)

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')          # Get specified environment object from workspace

#-----AUTOML_SETUP--------------------------------------------------------#
'''
Automated Machine Learning
* Enables to try multiple algorithms and preprocessing transformations
* By default, automated machine learning randomly selects from full range of algorithms for specified task
'''
# Retrieve metrics calculated by automated machine learning for a particular type of model task
for metric in automl_utils.get_primary_metrics('classification'):   # Get the primary metrics supported for a given task
    print(metric)

# Configure the automated machine learning run
automl_config = AutoMLConfig(                                   # Represents configuration for submitting an automated ML experiment in Azure ML
    name='Automated ML Experiment',                             # Name of the experiment
    task='classification',                                      # Type of task to run: 'classification', 'regression', or 'forecasting'
    compute_target=cluster_name,                                # Compute target to run the Automated ML experiment on
    training_data = train_ds,                                   # Training data to be used within the experiment
    validation_data = test_ds,                                  # Validation data to be used within the experiment
    label_column_name='Diabetic',                               # Name of the label column
    iterations=4,                                               # Total number of different algorithm and parameter combinations to test during an automated ML experiment
    primary_metric = 'AUC_weighted',                            # Metric that Automated Machine Learning will optimize for model selection
    max_concurrent_iterations=2,                                # Represents maximum number of iterations that would be executed in parallel
    featurization='auto'                                        # 'auto' / 'off' / FeaturizationConfig Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used
)

print("Ready for Auto ML run.")

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'mslearn-diabetes-mlflow'
automl_experiment = Experiment(                                 # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)
print('Submitting Auto ML experiment...')

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure ML
'''
# Submit an experiment incl config to be submitted and return the active created run
automl_run = automl_experiment.submit(                          # Run defines the base class for all Azure Machine Learning experiment runs
    config=automl_config,                                       # Config to be submitted
)                                   
print('Pipeline submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(pipeline_run).show()

automl_run.wait_for_completion()                                # Wait for the completion of this run, returns the status object after the wait
 
#-----PERFORMANCE--------------------------------------------------------------#
'''
Performance evaluation
* Metric for each run is logged
'''
# View all runs and their metrics
for run in automl_run.get_children():                           # Get all children for the current run selected by specified filters
    print('Run ID', run.id)
    for metric in run.get_metrics():                            # Retrieve the metrics logged to the run
        print('\t', run.get_metrics(metric))

# Get best run and respective model metrics
best_run, fitted_model = automl_run.get_output()                # Return the run with the corresponding best pipeline that has already been tested
print(best_run)

print('\nBest Model Definition:')
print(fitted_model)

print('\nBest Run Transformations:')
for step in fitted_model.named_steps:
    print(step)

print('\nBest Run Metrics:')
best_run_metrics = best_run.get_metrics()                       # Retrieve the metrics logged to the run
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)

#-----REGISTER_MODEL-----------------------------------------------------------#
'''
Register run machine learning model
* Outputs of the experiment also include the trained model file
* Register model in your Azure ML workspace
* Allowing to track model versions and retrieve them later
'''
best_run.register_model(                                        # Register a model for operationalization
    model_path='outputs/diabetes_model.pkl',                    # Relative cloud path to model
    model_name='diabetes_model',                                # Name of model
    tags={'Training context':'Auto ML'},                        # Dictionary of key value tags to assign to model
    properties={                                                # Dictionary of key value properties to assign to model, properties cannot be changed after model creation
        'AUC':  best_run_metrics['AUC'],
        'Accuracy': best_run_metrics['Accuracy']
    }
)
