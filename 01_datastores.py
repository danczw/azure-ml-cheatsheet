from azureml.core import Dataset, Workspace

#-----WORKSPACE----------------------------------------------------------------#
# load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----DATASTORE----------------------------------------------------------------#
'''
* references to storage locations
* e.g. Azure Storage blob containers
* Every workspace has a default datastore
* usually the Azure storage blob container that was created with the workspace
'''
# Get the default datastore
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print(ds_name, '- Default =', ds_name == default_ds.name)

#-----UPLOAD_DATA--------------------------------------------------------------#
default_ds.upload_files(
    files=['./data/sample_diabetes.csv'], # Upload the diabetes csv files in /data
    target_path='diabetes-data/', # Put it in a folder path in the datastore
    overwrite=True, # Replace existing files of the same name
    show_progress=True
)

#-----TABULAR_DATASET----------------------------------------------------------#
#Create a tabular dataset from the path on the datastore (this may take a short while)
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

# Display the first 20 rows as a Pandas dataframe
tab_data_set.take(20).to_pandas_dataframe()
print(tab_data_set)

#-----FILE_DATASET-------------------------------------------------------------#
#Create a file dataset from the path on the datastore (this may take a short while)
file_data_set = Dataset.File.from_files(path=(default_ds, 'diabetes-data/*.csv'))

# Get the files in the dataset
print(f'Files in datastore: {default_ds.name}')
for file_path in file_data_set.to_path():
    print(f'\t{file_path}')

#-----REGISTER_DATASET---------------------------------------------------------#
'''
register dataset to make them easily accessible to any experiment being run in the workspace
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