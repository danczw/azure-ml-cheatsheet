# Import libraries
from azureml.core import Environment, Experiment, Model, ScriptRunConfig, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import DockerConfiguration, RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline, ScheduleRecurrence, Schedule
from azureml.pipeline.core.run import PipelineRun
from azureml.pipeline.steps import PythonScriptStep
import requests
from azureml.widgets import RunDetails
import os

'''
Experiment
* Eexperiment is a named process, usually the running of a single script or a pipeline
    * Can generate metrics and outputs and be tracked in the Azure ML workspace
* Experiment can be run multiple times, with different data, code, or settings
* Azure ML tracks each run, enabling to view run history and compare results for each run

* This is the entry script for the pipeline, combining:
    * Uploaded dataset located in created datastore
    * Provisioned compute target
    * Defined Python environment
* Adds a data preprocessing and a model training stepwise in a pipeline for more flexibility
* Alternatively, for fast prototyping single script experiments can be run (see SCRIPT_SETUP)
'''

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig('prepped_data')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')

#-----SCRIPT_SETUP-------------------------------------------------------------#
'''
Single step experiments
* For running single step experiments, no pipeline need to be deployed
* Simply create a script run config - similar to a single pipeline step
    * Identifies the Python script file to be run in the experiment
    * Determines the compute target and Python environment
    * Creates a DockerConfiguration for the script run
    * Setting its use_docker attribute to True to host the script's environment in a Docker container
'''
# script_config = ScriptRunConfig(
#     source_directory=experiment_folder,
#     script='train_model.py',
#     arguments = [
#         '--regularization', 0.1,                        
#         '--input-data', diabetes_ds.as_named_input('training_data')],   # Reference to tabular dataset
#         # '--input-data', diabetes_ds.as_named_input('training_files').as_download()], # Reference to file dataset location
#     environment=registered_env,
#     compute_target=cluster_name,
#     docker_runtime_config=DockerConfiguration(use_docker=True)          # Use docker to host environment
# )
'''
Note: when using file dataset:
* Define path from which the script can read the files
* Either use as_download or as_mount method
    * as_download causes files in the file dataset to be downloaded to a temporary location on the compute where the script is being run
    * as_mount creates a mount point from which the files can be streamed directly from the datastore
'''

#-----PIPELINE_SETUP-----------------------------------------------------------#
'''
Azure Machine Learning Pipelines
* Consist of one or more steps
* Can be Python scripts, or specialized steps like a data transfer step copying data from one location to another
* Each step can run in its own compute context

This repo defines a simple pipeline containing two Python script steps:
* First step to pre-process training data
* Second step to use the pre-processed data to train a model
* Reuse is enabled:
    * Usually first step should run every time if the data has changed
    * Subsequent steps are triggered only if the output from step one changes
    * For convenience reuse enables to only run any steps with changed parameter

* Common kinds of step in an Azure Machine Learning pipeline:
    * PythonScriptStep: Runs specified Python script
    * DataTransferStep: Uses Azure Data Factory to copy data between data stores
    * DatabricksStep:   Runs notebook, script, or compiled JAR on a databricks cluster
    * AdlaStep:         Runs U-SQL job in Azure Data Lake Analytics
    * ParallelRunStep:  Runs Python script as a distributed task on multiple compute nodes.
'''
pipeline_run_config = RunConfiguration()                        # Create a new runconfig object for the pipeline
pipeline_run_config.target = cluster_name                       # Use the compute created  
pipeline_run_config.environment = registered_env                # Assign the environment to the run configuration

print ('Run configuration created.')

'''
Review ./experiments/* which includes both example pipeline steps
'''
experiment_folder = './experiments' # Pipeline steps folder

# Step 1, Run the data prep script
prep_step = PythonScriptStep(
    name = 'Prepare Data',                                      # Step name
    source_directory = experiment_folder,                       # Step py file location
    script_name = 'data_prep.py',                               # Step py file name
    arguments = [                                               # Experiment parameter 
        '--input-data', diabetes_ds.as_named_input('raw_data'), # Reference to tabular dataset
        '--prepped-data', prepped_data                          # Reference to output data
    ],                                                          
    compute_target = cluster_name,                              # Compute target
    runconfig = pipeline_run_config,                            # Pipeline config
    allow_reuse = True                                          # Reuse of previous calculations
)

# Step 2, run the training script
train_step = PythonScriptStep(
    name = 'Train and Register Model',                          # Step name
    source_directory = experiment_folder,                       # Step py file location
    script_name = 'train_model.py',                             # Step py file name
    arguments = [                                               # Experiment parameter 
        '--training-data', prepped_data.as_input(),             # Reference to step 1 output data
        '--regularization', 0.1                                 # Regularizaton rate parameter
    ],                                                          
    compute_target = cluster_name,                              # Compute target
    runconfig = pipeline_run_config,                            # Pipeline config
    allow_reuse = True                                          # Reuse of previous calculations
)

print('Pipeline steps defined')

# Construct the pipeline
pipeline_steps = [prep_step, train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print('Pipeline is built.')

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk'
experiment = Experiment(workspace=ws, name=experiment_name)
print('Pipeline submitted for execution.')

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure Machine Learning
'''
# run = experiment.submit(config=script_config)                       # Run single script experiment
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(pipeline_run).show()                                     # Show details

# run.wait_for_completion()                                           # Asynchronous - does not work with local execution
pipeline_run.wait_for_completion(show_output=True)

#-----LOGS---------------------------------------------------------------------#
# View run history
# diabetes_experiment = ws.experiments[experiment_name]
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

'''
Troubleshoot the experiment run
* Use get_details method to retrieve basic details about the run
* Use get_details_with_logs method to retrieve run details as well as contents of log files
'''
run_details = pipeline_run.get_details_with_logs()
# print(f'Run details: \n\t{run_details}')

# Download log files
log_folder = 'downloaded-logs'
pipeline_run.get_all_logs(destination=log_folder)
# Verify the files have been downloaded
for root, directories, filenames in os.walk(log_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

'''
Download the files produced by the experiment e.g. for logged visualizations
* Either individually by using the download_file method
* Or by using the download_files method to retrieve multiple files
'''
# Download 
download_folder = 'downloaded-files'
# Download files in the 'outputs' folder
pipeline_run.download_files(prefix='outputs', output_directory=download_folder)
# Verify the files have been downloaded
for root, directories, filenames in os.walk(download_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

#-----REGISTER_MODEL-----------------------------------------------------------#
'''
Register run mchine learning model
* Outputs of the experiment also include the trained model file
* Register model in your Azure Machine Learning workspace
* Allowing to track model versions and retrieve them later
'''
pipeline_run.register_model(
    model_path='outputs/diabetes_model.pkl',
    model_name='diabetes_model',
    tags={'Training context':'Script'},
    properties={
        'AUC': pipeline_run.get_metrics()['AUC'],
        'Accuracy': pipeline_run.get_metrics()['Accuracy']
    }
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

#-----ENDPOINT-----------------------------------------------------------------#
# Publish the pipeline from the run as a REST service
published_pipeline = pipeline_run.publish_pipeline(
    name='diabetes-training-pipeline', description='Trains diabetes model', version='1.0')

# Find its URI as a property of the published pipeline object
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

'''
Endpoint for model training calls
* To use an endpoint, client applications need to make a REST call over HTTP
* Request must be authenticated --> authorization header is required
* Real application would require a service principal with which to be authenticated
* For now, use the authorization header from the current connection to Azure workspace
'''
# Define authentication header
interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print('Authentication header ready.')

# Make REST call to get pipeline run ID
rest_endpoint = published_pipeline.endpoint
response = requests.post(
    rest_endpoint, 
    headers=auth_header, 
    json={'ExperimentName': experiment_name}
)
run_id = response.json()['Id']

# Use run ID to wait for pipeline to finish
published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
published_pipeline_run.wait_for_completion(show_output=True)

# Schedule pipeline e.g. for a weekly run
recurrence = ScheduleRecurrence(                                # Submit the Pipeline every Monday at 00:00 UTC
    frequency='Week',
    interval=1,
    week_days=['Monday'],
    time_of_day='00:00'
)
weekly_schedule = Schedule.create(                              # Schedule Pipeline
    ws, name='weekly-diabetes-training', 
    description='Based on time',
    pipeline_id=published_pipeline.id, 
    experiment_name='mslearn-diabetes-pipeline', 
    recurrence=recurrence
)
print('Pipeline scheduled.')

# List schedules
schedules = Schedule.list(ws)
schedules

# Get details of latest run
pipeline_experiment = ws.experiments.get(experiment_name)
latest_run = list(pipeline_experiment.get_runs())[0]
latest_run.get_details()