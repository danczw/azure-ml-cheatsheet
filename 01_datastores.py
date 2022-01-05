# Import libraries
from azureml.core import Dataset, Datastore, Workspace
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

#-----WORKSPACE----------------------------------------------------------------#
'''
Azure ML Workspace
* Defines the boundary for a set of related machine learning assets
* Group machine learning assets based on
    * Projects
    * Deployment environments (dev, test and production)
    * Teams
    * Other organizing principle
* Assign role-based authorization policies to a workspace
* --> enabling management of permissions that restrict what actions specific Azure Active Directory (AAD) principals can perform
'''
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASTORE----------------------------------------------------------------#
'''
Datastores
* References to storage locations, e.g.:
    * Azure Storage (blob and file containers)
    * Azure Data Lake stores
    * Azure SQL Database
    * Azure Databricks file system (DBFS) 
* Every workspace has a default datastore
    * Initially Azure storage blob container created with the workspace
'''

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:                                   # List of all datastores in the workspace
    print(ds_name, '- Default =', ds_name == ws.get_default_datastore().name)   # Gets default datastore for the workspace

# Register a new datastore
try:
    # Check if datastore is already existing
    ml_sdk_ds = Datastore.get(ws, datastore_name='ml_sdk_ds')   # Get a datastore by name - same as calling the constructor
except:
    # Create new datastore if it does not exist
    ml_sdk_ds = Datastore.register_azure_blob_container(        # Register an Azure Blob Container to the datastore
        workspace=ws,                                           # Target workspace
        datastore_name='ml_sdk_ds',                             # Name of datastore, case insensitive with only alphanumeric and _
        container_name='data-container',                        # Name of Azure blob container
        account_name=os.getenv('AZURE-ACCOUNT-NAME'),           # Storage account name
        account_key=os.getenv('AZURE-ACCOUNT-KEY')              # Access key to storage account
    )

# TODO: authentication failed when trying to connect to created datastore
# Set default store
ws.set_default_datastore('ml_sdk_ds')                           # Set the default datastore for the workspace
default_ds = ws.get_default_datastore()                         # Gets default datastore for the workspace, returns AzureBlobDatastore Class

#-----UPLOAD_DATA--------------------------------------------------------------#
# DEPRECIATED!
# Upload sample data to newly created blob storage
# default_ds.upload_files(                                        # Upload data from the local file system to blob container datastore points to
#     files=['./data/sample_diabetes.csv'],                       # List of absolute paths pf files to upload
#     target_path='diabetes-data/',                               # Location in blob container to upload the data to. Defaults to None, the root
#     overwrite=True,                                             # Indicates whether to overwrite existing files. Defaults to False
#     show_progress=True                                          # Indicates whether to show progress of upload in console. Defaults to True
# )

#-----TABULAR_DATASET----------------------------------------------------------#
# Create a tabular dataset from the path on the datastore (this may take a short while)
tab_data_set = Dataset.Tabular.from_delimited_files(            # Create a TabularDataset to represent tabular data in delimited files
    path=(default_ds, 'diabetes-data/*.csv')                    # Path to the source files, can be single value or list of http url string, DataPath object, or tuple of Datastore and relative path
)   

# Display the first 10 rows as a Pandas dataframe
tab_data_set.take(10).to_pandas_dataframe()                     # Take a sample of records from top of the dataset by the specified count
print(tab_data_set)

#-----FILE_DATASET-------------------------------------------------------------#
# Create a file dataset from the path on the datastore (this may take a short while)
file_data_set = Dataset.File.from_files(                        # Create a FileDataset to represent file streams
    path=(default_ds, 'diabetes-data/*.csv')                    # Path to the source files, can be single value or list of http url string, DataPath object, or tuple of Datastore and relative path
)

# Get the files in the dataset
print(f'Files in datastore: {default_ds.name}')
for file_path in file_data_set.to_path():                       # Get a list of file paths for each file stream defined by the dataset
    print(f'\t{file_path}')

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
for i in range(100):
    fname = str(i+1) + '.csv'
    sample[i].tofile(os.path.join(batch_folder, fname), sep=",")
print("Files saved!")

# Upload the files to the default datastore
default_ds.upload(                                              # Upload data from the local file system to blob container datastore points to
    src_dir="batch-data",                                       # The local directory to upload
    target_path="batch-data",                                   # Location in blob container to upload the data to. Defaults to None, the root
    overwrite=True,                                             # Indicates whether to overwrite existing files. Defaults to False
    show_progress=True                                          # Indicates whether to show progress of upload in console. Defaults to True.
)
print("Batch data uploaded!")

#-----REGISTER_DATASET---------------------------------------------------------#
'''
Register dataset for easy accessibility to any experiment run in the workspace
'''
# Register the tabular dataset
try:
    tab_data_set = tab_data_set.register(                       # Register the dataset to the provided workspace
        workspace=ws,                                           # The workspace to register the dataset
        name='diabetes dataset',                                # The name to register the dataset with
        description='diabetes data',                            # A text description of the dataset. Defaults to None
        tags = {'format':'CSV'},                                # Dictionary of key value tags to give the dataset. Defaults to None
        create_new_version=True                                 # Boolean to register the dataset as a new version under the specified name
    )
except Exception as ex:
    print(ex)

# Register the file dataset
try:
    file_data_set = file_data_set.register(                     # Register the dataset to the provided workspace
        workspace=ws,                                           # The workspace to register the dataset
        name='diabetes file dataset',                           # The name to register the dataset with
        description='diabetes files',                           # A text description of the dataset. Defaults to None
        tags = {'format':'CSV'},                                # Dictionary of key value tags to give the dataset. Defaults to None
        create_new_version=True                                 # Boolean to register the dataset as a new version under the specified name
    )
except Exception as ex:
    print(ex)

# Register batch dataset
batch_data_set = Dataset.File.from_files(                       # Create a FileDataset to represent file streams
    path=(default_ds, 'batch-data/'),                           # Path to the source files, can be single value or list of http url string, DataPath object, or tuple of Datastore and relative path
    validate=False                                              # Indicates whether to validate if data can be loaded from the returned dataset. Defaults to True
)
try:
    batch_data_set = batch_data_set.register(                   # Register the dataset to the provided workspace
        workspace=ws,                                           # The workspace to register the dataset
        name='batch-data',                                      # The name to register the dataset with
        description='batch data',                               # A text description of the dataset. Defaults to None
        create_new_version=True                                 # Boolean to register the dataset as a new version under the specified name
    )
except Exception as ex:
    print(ex)

print('Datasets registered')

# View registered datasets
print('Datasets:')
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print('\t', dataset.name, 'version', dataset.version)