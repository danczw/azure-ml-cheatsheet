# Import libraries
from azureml.core import Dataset, Datastore, Workspace

#-----WORKSPACE----------------------------------------------------------------#
'''
Azure Machine Learning Workspace
* Defines the boundary for a set of related machine learning assets
* Group machine learning assets based on
    * Projects
    * Deployment environments (dev, test and production)
    * Teams
    * Other organizing principle
* assign role-based authorization policies to a workspace
* --> enabling management of permissions that restrict what actions specific Azure Active Directory (AAD) principals can perform
'''
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

# view compute resources in workspace.compute_targets
# for compute_name in ws.compute_targets:
#     compute = ws.compute_targets[compute_name]
#     print('\t', compute.name, ':', compute.type)

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
# Get the default datastore
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print(ds_name, '- Default =', ds_name == default_ds.name)

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(
    workspace=ws, 
    datastore_name='blob_data', 
    container_name='data_container',
    account_name='YOUR-ACCOUNT-NAME',
    account_key='YOUR-AACOUNT-KEY'
)

# Get reference to new datastore
blob_store = Datastore.get(ws, datastore_name='blob_data')

# Set default store
ws.set_default_datastore('blob_data')
default_ds = ws.get_default_datastore() # reassign new default datastore variable

#-----UPLOAD_DATA--------------------------------------------------------------#
# Upload sample data to newly created blob storage
default_ds.upload_files(
    files=['./data/sample_diabetes.csv'],
    target_path='diabetes-data/',
    overwrite=True,
    show_progress=True
)

#-----TABULAR_DATASET----------------------------------------------------------#
# Create a tabular dataset from the path on the datastore (this may take a short while)
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

# Display the first 10 rows as a Pandas dataframe
tab_data_set.take(10).to_pandas_dataframe()
print(tab_data_set)

#-----FILE_DATASET-------------------------------------------------------------#
# Create a file dataset from the path on the datastore (this may take a short while)
file_data_set = Dataset.File.from_files(path=(default_ds, 'diabetes-data/*.csv'))

# Get the files in the dataset
print(f'Files in datastore: {default_ds.name}')
for file_path in file_data_set.to_path():
    print(f'\t{file_path}')

#-----REGISTER_DATASET---------------------------------------------------------#
'''
Register dataset for easy accessibility to any experiment run in the workspace
'''
# Register the tabular dataset
try:
    tab_data_set = tab_data_set.register(
        workspace=ws, 
        name='diabetes dataset',
        description='diabetes data',
        tags = {'format':'CSV'},
        create_new_version=True
    )
except Exception as ex:
    print(ex)

# Register the file dataset
try:
    file_data_set = file_data_set.register(
        workspace=ws,
        name='diabetes file dataset',
        description='diabetes files',
        tags = {'format':'CSV'},
        create_new_version=True
    )
except Exception as ex:
    print(ex)

print('Datasets registered')

# View registered datasets
print('Datasets:')
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print('\t', dataset.name, 'version', dataset.version)