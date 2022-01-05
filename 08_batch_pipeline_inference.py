# Import libraries
from azureml.core import Dataset, Environment, Experiment, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core.run import PipelineRun
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
import os
import pandas as pd
import requests

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the batch training dataset from registered datasets (see ./01_datastores.py)
batch_data_set = ws.datasets.get('batch-data')                  # Get specified dataset from list of all datasets in workspace

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
output_dir = OutputFileDatasetConfig(name='inferences')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')          # Get specified environment object from workspace
registered_env.base_image = DEFAULT_CPU_IMAGE                   # Base image used for Docker-based runs

#-----MODEL--------------------------------------------------------------------#
# Get the model - by default if model name is specified, latest version will be returned 
model = ws.models['diabetes_model']                             # # Get model by name from current workspace
print(model.name, 'version', model.version)

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

'''
ParallelRunStep
* ParallelRunStep which enables batch data parallel processing
* Results collated in single output file named parallel_run_step.txt
'''
# Define pipeline configuration
parallel_run_config = ParallelRunConfig(                        # Defines configuration for a ParallelRunStep object
    source_directory='./service',                               # Path to folder that contains the entry_script and supporting files used to execute on compute target
    entry_script="web_service_batch.py",                        # User script to be run in parallel on multiple nodes
    mini_batch_size="5",                                        # For TabularDataset input, this field is the approximate size of data the user script can process in one run() call
    error_threshold=10,                                         # number of record failures (TabularDataset) and file failures (FileDataset) ignored during processing
    output_action="append_row",                                 # How the output should be organized
    environment=registered_env,                                 # Environment definition that configures the Python environment
    compute_target=cluster_name,                                # Compute target to use for ParallelRunStep execution
    node_count=2                                                # Number of nodes in the compute target used for running the ParallelRunStep
)

# Define pipeline parallel step
parallelrun_step = ParallelRunStep(                             # Creates an Azure Machine Learning Pipeline step to process large amounts of data asynchronously and in parallel
    name='batch-score-diabetes',                                # Name of the step
    parallel_run_config=parallel_run_config,                    # A ParallelRunConfig object used to determine required run properties
    inputs=[batch_data_set.as_named_input('diabetes_batch')],   # List of input datasets
    output=output_dir,                                          # Output port binding, may be used by later pipeline steps
    arguments=[],                                               # List of command-line arguments to pass to the Python entry_script
    allow_reuse=True                                            # Whether the step should reuse previous results when run with the same settings/inputs
)

print('Steps defined')

# Construct the pipeline
pipeline = Pipeline(                                            # Create and manage workflows that stitch together various machine learning phases
    workspace=ws,                                               # Workspace to submit the Pipeline on
    steps=[parallelrun_step]                                    # List of steps to execute as part of a Pipeline
)

print('Pipeline is built.')

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk-batch'
experiment = Experiment(                                        # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)
print('Pipeline submitted for execution.')

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
# Get the run for the first and only step and download its output
prediction_run = next(pipeline_run.get_children())              # Get all children for the current run selected by specified filters
prediction_output = prediction_run.get_output_data('inferences')    # Get the output data from a given output
prediction_output.download(local_path='diabetes-results')       # Download the data represented by the PortDataReference

# Traverse the folder hierarchy and find the results file
for root, dirs, files in os.walk('diabetes-results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# Cleanup output format
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]

# Display the first 20 results
# print(df.head(20))

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
    name='diabetes-batch-pipeline',                             # Name of the published pipeline
    description='Batch scoring of diabetes data',               # Description of the published pipeline
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
