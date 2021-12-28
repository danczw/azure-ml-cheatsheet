from azureml.core import Environment, Experiment, Model, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails
import os

#-----WORKSPACE----------------------------------------------------------------#
# load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, "loaded")

# view compute resources in workspace.compute_targets
# for compute_name in ws.compute_targets:
#     compute = ws.compute_targets[compute_name]
#     print("\t", compute.name, ':', compute.type)

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get("diabetes dataset")

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./02_envs.py)
registered_env = Environment.get(ws, 'experiment_env')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./03_compute.py)
cluster_name = "ml-sdk-cc"

#-----EXPERIMENT_CONFIG--------------------------------------------------------#
"""
Create a script run config
* identifying the Python script file to be run in the experiment
* also determines the compute target and Python environment
* creates a DockerConfiguration for the script run
    * setting its use_docker attribute to True in order to host the script's environment in a Docker container
    * default behavior and can be omitted
"""
experiment_folder = './experiments'

# Create a script config
script_config = ScriptRunConfig(
    source_directory=experiment_folder,
    script='experiment_script.py',
    arguments = [
        '--regularization', 0.1, # Regularizaton rate parameter
        '--input-data', diabetes_ds.as_named_input('training_data')], # Reference to tabular dataset
        # '--input-data', diabetes_ds.as_named_input('training_files').as_download()], # Reference to file dataset location
    environment=registered_env,
    compute_target=cluster_name,
    docker_runtime_config=DockerConfiguration(use_docker=True) # Use docker to host environment
)
"""
when using file dataset:
* define path from which the script can read the files
* either use as_download or as_mount method
    * as_download causes files in the file dataset to be downloaded to a temporary location on the compute where the script is being run
    * as_mount creates a mount point from which the files can be streamed directly from the datastore
"""

#-----EXPERIMENT---------------------------------------------------------------#
experiment = Experiment(workspace=ws, name='ml-sdk')                # Create an Azure ML experiment in your workspace

#-----RUN----------------------------------------------------------------------#
# Run object is a reference to an individual run of an experiment in Azure Machine Learning
run = experiment.submit(config=script_config)                       # Run experiment

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(run).show()                                              # Show details

run.wait_for_completion()                                           # Asynchronous - does not work with local execution

#-----LOGS---------------------------------------------------------------------#
# View run history
# diabetes_experiment = ws.experiments['ml-sdk']
# for logged_run in diabetes_experiment.get_runs():                   # Iterate through runs
#     print('Run ID:', logged_run.id)
#     metrics = logged_run.get_metrics()
#     for key in metrics.keys():
#         print('-', key, metrics.get(key))

# Get logged metrics
# metrics = run.get_metrics()
# for key in metrics.keys():
#         print(key, metrics.get(key))
# print('\n')
# for file in run.get_file_names():
#     print(file)

"""
troubleshoot the experiment run
* use get_details method to retrieve basic details about the run
* or use get_details_with_logs method to retrieve run details as well as contents of log files
"""
run_details = run.get_details_with_logs()
# print(f"Run details: \n\t{run_details}")

# Download log files
log_folder = 'downloaded-logs'
run.get_all_logs(destination=log_folder)
# Verify the files have been downloaded
for root, directories, filenames in os.walk(log_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

"""
download the files produced by the experiment e.g. for logged visualizations
* either individually by using the download_file method
* or by using the download_files method to retrieve multiple files
"""
# Download 
download_folder = 'downloaded-files'
# Download files in the "outputs" folder
run.download_files(prefix='outputs', output_directory=download_folder)
# Verify the files have been downloaded
for root, directories, filenames in os.walk(download_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

#-----REGISTER_MODEL-----------------------------------------------------------#
"""
* outputs of the experiment include the trained model file
* register model in your Azure Machine Learning workspace
* allowing to track model versions and retrieve them later
"""
run.register_model(
    model_path='outputs/diabetes_model.pkl',
    model_name='diabetes_model',
    tags={'Training context':'Script'},
    properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']}
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