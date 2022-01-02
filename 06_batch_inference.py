# Import libraries
from azureml.core import Dataset, Environment, Experiment, Workspace
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
import os
import pandas as pd

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')

#-----MODEL--------------------------------------------------------------------#
# Get the model - by default if model name is specified, latest version will be returned 
model = ws.models['diabetes_model']
print(model.name, 'version', model.version)

#-----BATCH_DATA---------------------------------------------------------------#
'''
Creating batch data
* For batch inference purposes, batch data needs to be created
'''
# Load the diabetes data
diabetes = pd.read_csv('data/sample_diabetes.csv')
# Get a 100-item sample of the feature columns (not the diabetic label)
sample = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].sample(n=100).values

# Create a folder
batch_folder = './batch-data'
os.makedirs(batch_folder, exist_ok=True)
print("Folder created!")

# Save each sample as a separate file
print("Saving files...")
for i in range(100):
    fname = str(i+1) + '.csv'
    sample[i].tofile(os.path.join(batch_folder, fname), sep=",")
print("files saved!")

# Upload the files to the default datastore
print("Uploading files to datastore...")
default_ds = ws.get_default_datastore()
default_ds.upload(src_dir="batch-data", target_path="batch-data", overwrite=True, show_progress=True)

# Register a dataset for the input data
batch_data_set = Dataset.File.from_files(path=(default_ds, 'batch-data/'), validate=False)
try:
    batch_data_set = batch_data_set.register(
        workspace=ws, 
        name='batch-data',
        description='batch data',
        create_new_version=True
    )
except Exception as ex:
    print(ex)

print("Batch data uploaded!")

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
output_dir = OutputFileDatasetConfig(name='inferences')

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
# Review ./experiments/* which includes example pipeline steps
experiment_folder = './experiments' # Pipeline steps folder

parallel_run_config = ParallelRunConfig(
    source_directory=experiment_folder,
    entry_script="batch_diabetes.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=registered_env,
    compute_target=inference_cluster,
    node_count=2)

parallelrun_step = ParallelRunStep(
    name='batch-score-diabetes',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('diabetes_batch')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)

print('Steps defined')

pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
pipeline_run = Experiment(ws, 'mslearn-diabetes-batch').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)