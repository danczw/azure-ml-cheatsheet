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
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the batch training dataset from registered datasets (see ./01_datastores.py)
batch_data_set = ws.datasets.get('batch-data')

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
output_dir = OutputFileDatasetConfig(name='inferences')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')
registered_env.base_image = DEFAULT_CPU_IMAGE

#-----MODEL--------------------------------------------------------------------#
# Get the model - by default if model name is specified, latest version will be returned 
model = ws.models['diabetes_model']
print(model.name, 'version', model.version)

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

'''
ParallelRunStep
* ParallelRunStep which enables batch data parallel processing
* Results collated in single output file named parallel_run_step.txt
'''
# Define pipeline configuration
parallel_run_config = ParallelRunConfig(                        # Create a new runconfig object for the pipeline
    source_directory='./service',                               # Web service entry script location 
    entry_script="web_service.batch.py",                        # Web service entry script name
    mini_batch_size="5",                                        # Batch size
    error_threshold=10,                                         # number of record failures (TabularDataset) and file failures (FileDataset) ignored during processing
    output_action="append_row",                                 # output organization
    environment=registered_env,                                 # Inference env
    compute_target=cluster_name,                                # Compute target
    node_count=2                                                # Nodes in compute target used for parallel processing
)

parallelrun_step = ParallelRunStep(
    name='batch-score-diabetes',                                # Step name
    parallel_run_config=parallel_run_config,                    # Pipeline config
    inputs=[batch_data_set.as_named_input('diabetes_batch')],   # Input data
    output=output_dir,                                          # Output directory
    arguments=[],                                               # Experiment parameter
    allow_reuse=True                                            # Reuse of previous calculations
)

print('Steps defined')

# Construct the pipeline
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])

#-----EXPERIMENT_&_RUN---------------------------------------------------------#
# Create an Azure ML experiment in workspace and submit a run
experiment_name = 'mslearn-diabetes-batch'
pipeline_run = Experiment(ws, 'mslearn-diabetes-batch').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

#-----LOGS---------------------------------------------------------------------#
# Get the run for the first and only step and download its output
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='diabetes-results')

# Traverse the folder hierarchy and find the results file
for root, dirs, files in os.walk('diabetes-results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# cleanup output format
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
published_pipeline = pipeline_run.publish_pipeline(
    name='diabetes-batch-pipeline', description='Batch scoring of diabetes data', version='1.0')

# Find its URI as a property of the published pipeline object
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

# Define authentication header
interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print('Authentication header ready.')

# Make REST call to get pipeline run ID
rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "mslearn-diabetes-batch"})
run_id = response.json()["Id"]

# Use run ID to wait for pipeline to finish
published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
published_pipeline_run.wait_for_completion(show_output=True)

# Get details of latest run
pipeline_experiment = ws.experiments.get(experiment_name)
latest_run = list(pipeline_experiment.get_runs())[0]
latest_run.get_details()
