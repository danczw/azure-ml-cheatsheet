# Import libraries
from azureml.core import Experiment, Environment, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')               # Get specified dataset from list of all datasets in workspace

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')          # Get specified environment object from workspace

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
experiment_folder = './experiments'                             # Experiment script folder

# Create a script config
script_mlflow = ScriptRunConfig(                                # Represents configuration information for submitting a training run in Azure Machine Learning
    source_directory=experiment_folder,                         # Local directory containing code files needed for a run
    script='05_experiment_script_mlflow.py',                    # File path relative to the source_directory of the script to be run
    arguments = [                                               # Optional command line arguments to pass to the training script
        '--input-data', diabetes_ds.as_named_input('raw_data')  # Reference to tabular dataset
        # , '--input-data', diabetes_ds.as_named_input('training_files').as_download()    # Reference to file dataset location
    ],
    environment=registered_env,                                 # Environment to use for run, if no environment is specified, azureml.core.runconfig.DEFAULT_CPU_IMAGE will be used as the Docker image for the run 
    compute_target=cluster_name,                                # Compute target where training will happen, can be ComputeTarget object, existing ComputeTarget name, or the string "local" (default)
    docker_runtime_config=DockerConfiguration(use_docker=True)  # For jobs that require Docker runtime-specific configurations
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

For MLFlow logging see ./experiments/05_single_experiment_mlflow
'''
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk-experiment-mlflow'
experiment = Experiment(                                        # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)

#-----RUN----------------------------------------------------------------------#
# Submit an experiment incl config to be submitted and return the active created run
run = experiment.submit(config=script_mlflow)                   # Run defines the base class for all Azure Machine Learning experiment runs                
print('Experiment submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(run).show()

run.wait_for_completion()                                       # Wait for the completion of this run, returns the status object after the wait


#-----LOGS---------------------------------------------------------------------#
# Get logged metrics
metrics = run.get_metrics()                                     # Retrieve the metrics logged to the run
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')