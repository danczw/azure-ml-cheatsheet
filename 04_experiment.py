# Import libraries
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails
import os

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

Note: when using file dataset:
* Define path from which the script can read the files
* Either use as_download or as_mount method
    * as_download causes files in the file dataset to be downloaded to a temporary location on the compute where the script is being run
    * as_mount creates a mount point from which the files can be streamed directly from the datastore
'''

# Review ./experiments/* which includes single experiment file
experiment_folder = './experiments'                             # Experiment script folder

script_config = ScriptRunConfig(                                # Represents configuration information for submitting a training run in Azure Machine Learning
    source_directory=experiment_folder,                         # Local directory containing code files needed for a run
    script='04_experiment_script.py',                           # File path relative to the source_directory of the script to be run
    arguments = [                                               # Optional command line arguments to pass to the training script
        '--input-data', diabetes_ds.as_named_input('raw_data')  # Reference to tabular dataset
        # , '--input-data', diabetes_ds.as_named_input('training_files').as_download()    # Reference to file dataset location
    ],
    environment=registered_env,                                 # Environment to use for run, if no environment is specified, azureml.core.runconfig.DEFAULT_CPU_IMAGE will be used as the Docker image for the run                      
    compute_target=cluster_name,                                # Compute target where training will happen, can be ComputeTarget object, existing ComputeTarget name, or the string "local" (default)
    docker_runtime_config=DockerConfiguration(use_docker=True)  # For jobs that require Docker runtime-specific configurations
)

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk-experiment'
experiment = Experiment(                                        # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure ML
'''
# Submit an experiment incl config to be submitted and return the active created run
run = experiment.submit(config=script_config)                   # Run defines the base class for all Azure Machine Learning experiment runs                
print('Experiment submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(run).show()

run.wait_for_completion()                                       # Wait for the completion of this run, returns the status object after the wait

#-----LOGS---------------------------------------------------------------------#
# View run history
diabetes_experiment = ws.experiments[experiment_name]           # Retrieve an experiment
for logged_run in diabetes_experiment.get_runs():               # Return a generator of the runs for this experiment, in reverse chronological order
    print('Run ID:', logged_run.id)
    metrics = logged_run.get_metrics()                          # Retrieve the metrics logged to the run
    for key in metrics.keys():
        print('-', key, metrics.get(key))

# Get logged metrics
metrics = run.get_metrics()                                     # Retrieve the metrics logged to the run
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')

# Get logged files
for file in run.get_file_names():                               # List the files that are stored in association with the run
    print(file)

#-----TROUBLESHOOT-------------------------------------------------------------#
'''
Troubleshoot the experiment run
* Use get_details method to retrieve basic details about the run
* Use get_details_with_logs method to retrieve run details as well as contents of log files
'''
run_details = run.get_details_with_logs()                       # Return run status including log file content
print(f'Run details: \n\t{run_details}')

# Download log files
log_folder = 'downloaded-logs'
run.get_all_logs(destination=log_folder)                        # Download all logs for the run to a directory
# Verify the files have been downloaded
for root, directories, filenames in os.walk(log_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))