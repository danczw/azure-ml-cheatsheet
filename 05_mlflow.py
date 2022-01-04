# Import libraries
from azureml.core import Experiment, Environment, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')

#-----SCRIPT_SETUP-------------------------------------------------------------#
'''
Experiment script
* Experiment is a named process, usually the running of a single script or a pipeline
* Can generate metrics and outputs and be tracked in the Azure ML workspace
* Experiment can be run multiple times, with different data, code, or settings
* Azure ML tracks each run, enabling to view run history and compare results for each run
* Single step experiments
    * For running single step experiments, no pipeline need to be deployed
    * Create a script run config - similar to a single pipeline step (see 05_pipeline.py)
        * Identifies the Python script file to be run in the experiment
        * Determines the compute target and Python environment
        * Creates a DockerConfiguration for the script run
'''
# Review ./experiments/* which includes single experiment file
experiment_folder = './experiments'                                 # Experiment script folder

# Create a script config
script_mlflow = ScriptRunConfig(
    source_directory=experiment_folder,
    script='05_experiment_script_mlflow.py',
    arguments = [                      
        '--input-data', diabetes_ds.as_named_input('raw_data')      # Reference to tabular dataset
    ],
    environment=registered_env,
    compute_target=cluster_name,
    docker_runtime_config=DockerConfiguration(use_docker=True)      # Use docker to host environment
)

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

Note: for MLFlow logging see ./experiments/05_single_experiment_mlflow
'''
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk-experiment-mlflow'
experiment = Experiment(workspace=ws, name=experiment_name)

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure Machine Learning
'''
run = experiment.submit(config=script_mlflow)
print('Experiment submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(run).show()

run.wait_for_completion()

#-----LOGS---------------------------------------------------------------------#
# Get logged metrics
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')