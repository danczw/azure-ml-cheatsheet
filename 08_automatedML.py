# Import libraries
from azureml.core import Environment, Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
import azureml.train.automl.utilities as automl_utils
from azureml.widgets import RunDetails

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')

# Split for train and test dataset
train_ds, test_ds = diabetes_ds.random_split(percentage=0.7, seed=123)

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')

#-----AUTOML_SETUP--------------------------------------------------------#
'''
Automated Machine Learning
* Enables to try multiple algorithms and preprocessing transformations
* By default, automated machine learning randomly selects from full range of algorithms for specified task
'''
# Retrieve metrics calculated by automated machine learning for a particular type of model task
for metric in automl_utils.get_primary_metrics('classification'):
    print(metric)

# Configure the automated machine learning run
automl_config = AutoMLConfig(
    name='Automated ML Experiment',                             # 
    task='classification',                                      # Type of task to run: 'classification', 'regression', or 'forecasting'
    compute_target=cluster_name,                                # Compute target
    training_data = train_ds,                                   # Training dataset
    validation_data = test_ds,                                  # Test dataset
    label_column_name='Diabetic',                               # Name of label column
    iterations=4,                                               # Total number of different algorithm and parameter combinations
    primary_metric = 'AUC_weighted',                            # Metric that Automated ML will optimize for model selection
    max_concurrent_iterations=2,                                # Maximum number of iterations that would be executed in parallel
    featurization='auto'                                        # Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used
)

print("Ready for Auto ML run.")


#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'mslearn-diabetes-hyperdrive'

print('Submitting Auto ML experiment...')
automl_experiment = Experiment(ws, experiment_name)


#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure Machine Learning
'''
automl_run = automl_experiment.submit(automl_config)

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(pipeline_run).show()

automl_run.wait_for_completion(show_output=True)

#-----PERFORMANCE--------------------------------------------------------------#
'''
Performance evaluation
* Metric for each run is logged
'''
# View all runs and their metrics
for run in automl_run.get_children():
    print('Run ID', run.id)
    for metric in run.get_metrics():
        print('\t', run.get_metrics(metric))

# Get best run and respective model metrics
best_run, fitted_model = automl_run.get_output()
print(best_run)
print('\nBest Model Definition:')
print(fitted_model)
print('\nBest Run Transformations:')
for step in fitted_model.named_steps:
    print(step)
print('\nBest Run Metrics:')
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)

from azureml.core import Model


#-----REGISTER_MODEL-----------------------------------------------------------#
'''
Register run machine learning model
* Outputs of the experiment also include the trained model file
* Register model in your Azure Machine Learning workspace
* Allowing to track model versions and retrieve them later
'''
best_run.register_model(
    model_path='outputs/model.pkl',
    model_name='diabetes_model',
    tags={'Training context':'Auto ML'},
    properties={'AUC': best_run_metrics['AUC_weighted'],
    'Accuracy': best_run_metrics['accuracy']}
)

# List registered models
# for model in Model.list(ws):
#     print(model.name, 'version:', model.version)
#     for tag_name in model.tags:
#         tag = model.tags[tag_name]
#         print ('\t',tag_name, ':', tag)
#     for prop_name in model.properties:
#         prop = model.properties[prop_name]
#         print ('\t',prop_name, ':', prop)
#     print('\n')
