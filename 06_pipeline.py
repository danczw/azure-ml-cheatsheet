# Import libraries
from azureml.core import Environment, Experiment, Model, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline, ScheduleRecurrence, Schedule
from azureml.pipeline.core.run import PipelineRun
from azureml.pipeline.steps import PythonScriptStep
from azureml.widgets import RunDetails
import os
import requests

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

#-----PIPELINE_SETUP-----------------------------------------------------------#
'''
Azure ML Pipelines
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

* Common step types in an Azure ML pipeline:
    * PythonScriptStep: Runs specified Python script
    * DataTransferStep: Uses Azure Data Factory to copy data between data stores
    * DatabricksStep:   Runs notebook, script, or compiled JAR on a databricks cluster
    * AdlaStep:         Runs U-SQL job in Azure Data Lake Analytics
    * ParallelRunStep:  Runs Python script as a distributed task on multiple compute nodes.
'''
# Define pipeline configuration
pipeline_run_config = RunConfiguration()                        # Represents configuration for experiment runs targeting different compute targets in Azure ML
pipeline_run_config.target = cluster_name                       # Set compute target where job is scheduled for execution
pipeline_run_config.environment = registered_env                # Environment definition: assign the environment to the run configuration
print ('Run configuration created.')

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig('prepped_data')          # Represent how to copy the output of a run and be promoted as a FileDataset

# Review ./experiments/* which includes example pipeline steps
experiment_folder = './experiments'                             # Experiment script folder

# Step 1, Run the data prep script
prep_step = PythonScriptStep(
    name = 'Prepare Data',                                      # Name of the step
    source_directory = experiment_folder,                       # Folder that contains Python script, conda env, and other resources used in the step
    script_name = '06_data_prep.py',                            # Name of a Python script relative to source_directory
    arguments = [                                               # Command line arguments for the Python script file, arguments will be passed to compute via arguments parameter in RunConfiguration 
        '--input-data', diabetes_ds.as_named_input('raw_data'), # Reference to tabular dataset
        '--prepped-data', prepped_data                          # Reference to output data
    ],                                                          
    compute_target = cluster_name,                              # Compute target to use
    runconfig = pipeline_run_config,                            # RunConfiguration to specify additional requirements for run, such as conda dependencies and a docker image
    allow_reuse = True                                          # Indicates whether the step should reuse previous results when re-run with the same settings
)

# Step 2, run the training script
train_step = PythonScriptStep(
    name = 'Train and Register Model',                          # Name of the step
    source_directory = experiment_folder,                       # Folder that contains Python script, conda env, and other resources used in the step
    script_name = '06_train_model.py',                          # Name of a Python script relative to source_directory
    arguments = [                                               # Command line arguments for the Python script file, arguments will be passed to compute via arguments parameter in RunConfiguration 
        '--training-data', prepped_data.as_input(),             # Reference to step 1 output data
        '--regularization', 0.1                                 # Regularizaton rate parameter
    ],                                                          
    compute_target = cluster_name,                              # Compute target to use
    runconfig = pipeline_run_config,                            # RunConfiguration to specify additional requirements for run, such as conda dependencies and a docker image
    allow_reuse = True                                          # Indicates whether the step should reuse previous results when re-run with the same settings
)

print('Pipeline steps defined')

# Construct the pipeline
pipeline_steps = [prep_step, train_step]
pipeline = Pipeline(                                            # Create and manage workflows that stitch together various machine learning phases
    workspace=ws,                                               # Workspace to submit the Pipeline on
    steps=pipeline_steps                                        # List of steps to execute as part of a Pipeline
)
print('Pipeline is built.')

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk-pipeline'
experiment = Experiment(                                        # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure ML
'''
# Submit an experiment incl config to be submitted and return the active created run
pipeline_run = experiment.submit(                               # Run defines the base class for all Azure Machine Learning experiment runs
    pipeline,                                                   # Config to be submitted
    regenerate_outputs=True                                     # Whether to force regeneration of all step outputs and disallow data reuse for run, default is False
)                                   
print('Pipeline submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(pipeline_run).show()

pipeline_run.wait_for_completion()                              # Wait for the completion of this run, returns the status object after the wait

#-----LOGS---------------------------------------------------------------------#
# Review metrics for each step
for run in pipeline_run.get_children():                         # Get all children for the current run selected by specified filters
    print(run.name, ':')
    metrics = run.get_metrics()                                 # Retrieve the metrics logged to the run
    for metric_name in metrics:
        print('\t',metric_name, ":", metrics[metric_name])

#-----TROUBLESHOOT-------------------------------------------------------------#
'''
Troubleshoot the experiment run
* Use get_details method to retrieve basic details about the run
* Use get_details_with_logs method to retrieve run details as well as contents of log files
'''
run_details = pipeline_run.get_details_with_logs()              # Return run status including log file content
print(f'Run details: \n\t{run_details}')

# Download log files
log_folder = 'downloaded-logs'
pipeline_run.get_all_logs(destination=log_folder)               # Download all logs for the run to a directory
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
pipeline_run.download_files(                                    # Download files from a given storage prefix (folder name) or entire container if prefix is unspecified
    prefix='outputs',                                           # Filepath prefix within container from which to download all artifacts
    output_directory=download_folder                            # Optional directory that all artifact paths use as a prefix
)

# Verify the files have been downloaded
for root, directories, filenames in os.walk(download_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

#-----REGISTER_MODEL-----------------------------------------------------------#
'''
Register run machine learning model
* Outputs of the experiment also include the trained model file
* Register model in your Azure ML workspace
* Allowing to track model versions and retrieve them later
'''
pipeline_run.register_model(                                    # Register a model for operationalization
    model_path='outputs/diabetes_model.pkl',                    # Relative cloud path to model
    model_name='diabetes_model',                                # Name of model
    tags={'Training context':'Script'},                         # Dictionary of key value tags to assign to model
    properties={                                                # Dictionary of key value properties to assign to model, properties cannot be changed after model creation
        'AUC': pipeline_run.get_metrics()['AUC'],
        'Accuracy': pipeline_run.get_metrics()['Accuracy']
    }
)

# List registered models - training concept of latest model should be 'pipeline'
for model in Model.list(ws):                                    # Retrieve a list of all models associated with the provided workspace, with optional filters 
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')

#-----ENDPOINT-----------------------------------------------------------------#
'''
Endpoint for model training calls
* To use an endpoint, client applications need to make a REST call over HTTP
* Request must be authenticated --> authorization header is required
* Real application would require a service principal with which to be authenticated
* For now, use the authorization header from the current connection to Azure workspace
'''
# Publish the pipeline from the run as a REST service
published_pipeline = pipeline_run.publish_pipeline(             # Publish a pipeline and make it available for rerunning
    name='diabetes-training-pipeline',                          # Name of the published pipeline
    description='Trains diabetes model',                        # Description of the published pipeline
    version='1.0'                                               # Version of the published pipeline
)

# Find its URI as a property of the published pipeline object
rest_endpoint = published_pipeline.endpoint                     # REST endpoint URL to submit runs for this pipeline
print(rest_endpoint)

# Define authentication header
interactive_auth = InteractiveLoginAuthentication()             # Manages authentication and acquires an authorization token in interactive login workflows
auth_header = interactive_auth.get_authentication_header()      # Return the HTTP authorization header, authorization header contains the user access token for access authorization against the service
print('Authentication header ready.')

# Make REST call to get pipeline run ID
response = requests.post(
    rest_endpoint, 
    headers=auth_header, 
    json={'ExperimentName': experiment_name}
)
run_id = response.json()['Id']

# Use run ID to wait for pipeline to finish
published_pipeline_run = PipelineRun(                           # Represents a run of a Pipeline
    ws.experiments[experiment_name],                            # Experiment object associated with the pipeline run
    run_id                                                      # Run ID of the pipeline run
)
published_pipeline_run.wait_for_completion()                    # Wait for the completion of this run, returns the status object after the wait

# Get details of latest run
pipeline_experiment = ws.experiments.get(experiment_name)       # Get experiment by name of current workspace
latest_run = list(pipeline_experiment.get_runs())[0]            # Return a generator of the runs for this experiment, in reverse chronological order
latest_run.get_details()                                        # Get the definition, status information, current log files, and other details of the run

#-----SCHEDULE-----------------------------------------------------------------#
# Schedule pipeline e.g. for a weekly run
recurrence = ScheduleRecurrence(                                # Defines the frequency, interval and start time of a pipeline Schedule
    frequency='Week',                                           # The unit of time that describes how often schedule fires: "Minute", "Hour", "Day", "Week", or "Month"
    interval=1,                                                 # Value specifying how often schedule fires based on frequency, which is the number of time units to wait until the schedule fires again
    week_days=['Monday'],                                       # If frequency is "Week", specify one or more days, separated by commas, when to run the workflow    
    time_of_day='00:00'                                         # If frequency is "Day" or "Week", specify a time of day for schedule to run as a string in the form hh:mm
)
weekly_schedule = Schedule.create(                              # Create a schedule for a pipeline
    ws,                                                         # Workspace object this Schedule will belong to
    name='weekly-diabetes-training',                            # Name of the schedule
    description='Based on time',                                # Description of the schedule
    pipeline_id=published_pipeline.id,                          # ID of the pipeline the schedule will submit
    experiment_name='mslearn-diabetes-pipeline',                # Name of the experiment schedule will submit runs on
    recurrence=recurrence                                       # Schedule recurrence of the pipeline
)
print('Pipeline scheduled.')

# List schedules
schedules = Schedule.list(ws)                                   # Get all schedules in the current workspace
schedules
