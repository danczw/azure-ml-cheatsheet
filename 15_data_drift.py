# Import libraries
from azureml.core import Datastore, Dataset, Workspace
from azureml.datadrift import DataDriftDetector
from azureml.widgets import RunDetails
import datetime as dt
import pandas as pd

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----BASELINE_DATASET---------------------------------------------------------#
'''
Baseline dataset
* To monitor a dataset for data drift, register a baseline dataset
* (Usually the dataset used to train model)
* Used as a point of comparison with data collected in the future
'''
# Upload the baseline data
default_ds = ws.get_default_datastore()                         # Get the default datastore for the workspace
default_ds.upload_files(
    files=['./data/sample_diabetes.csv'],
    target_path='diabetes-baseline',
    overwrite=True, 
    show_progress=True
)

# Create and register the baseline dataset
print('Registering baseline dataset...')
baseline_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-baseline/*.csv'))
baseline_data_set = baseline_data_set.register(
    workspace=ws, 
    name='diabetes baseline',
    description='diabetes baseline data',
    tags = {'format':'CSV'},
    create_new_version=True
)

print('Baseline dataset registered!')

#-----TARGET_DATASET-----------------------------------------------------------#
'''
Target dataset
* Over time, new data might be collected with the same features as your baseline training data
* To compare this new data to the baseline data, define a target dataset
    * Must includes features to be analyzed for data drift
    * Must includes timestamp field indicating point in time when the new data was current
    * Timestamp can either be a field in the dataset itself, or derived from the folder and filename pattern used to store the data.
    * Example:
        * Store new data in a folder hierarchy that consists of a folder for the year
        * Each yer folder contains a folder for the month
        * Which in turn contains a folder for the day
        * Or encode the year, month, and day in the file name: 'data_2020-01-29.csv'
    * --> Enables measurement of data drift over temporal intervals
'''
print('Generating simulated data...')

# Load data file
data = pd.read_csv('data/sample_diabetes.csv')

# Generate data for the past 6 weeks
weeknos = reversed(range(6))

file_paths = []
for weekno in weeknos:
    
    # Get the date X weeks ago
    data_date = dt.date.today() - dt.timedelta(weeks=weekno)
    
    # Modify data to ceate some drift
    data['Pregnancies'] = data['Pregnancies'] + 1
    data['Age'] = round(data['Age'] * 1.2).astype(int)
    data['BMI'] = data['BMI'] * 1.1
    
    # Save the file with the date encoded in the filename
    file_path = 'data/diabetes_{}.csv'.format(data_date.strftime("%Y-%m-%d"))
    data.to_csv(file_path)
    file_paths.append(file_path)

# Upload the files
path_on_datastore = 'diabetes-target'
default_ds.upload_files(
    files=file_paths,
    target_path=path_on_datastore,
    overwrite=True,
    show_progress=True
)

# Use the folder partition format to define a dataset with a 'date' timestamp column
partition_format = path_on_datastore + '/diabetes_{date:yyyy-MM-dd}.csv'
target_data_set = Dataset.Tabular.from_delimited_files(
    path=(default_ds, path_on_datastore + '/*.csv'),
    partition_format=partition_format
)

# Register the target dataset
print('Registering target dataset...')
target_data_set = target_data_set.with_timestamp_columns('date').register(
    workspace=ws,
    name='diabetes target',
    description='diabetes target data',
    tags = {'format':'CSV'},
    create_new_version=True
)

print('Target dataset registered!')

#-----DRIFT_MONITOR------------------------------------------------------------#
'''
Data drift monitor
* Specify a schedule on which it should run
    * Frequency: define a schedule to run every Day, Week, or Month
	* Latency: number of hours to allow for new data to be collected and added to the target dataset
* Additionally, specify threshold for the rate of data drift and an operator email address for notifications if threshold is exceeded
	* Data drift is measured using a calculated magnitude of change in the statistical distribution of feature values over time
    * Threshold: data drift magnitude above which you want to be notified
'''
# set up feature list to be monitored
features = ['Pregnancies', 'Age', 'BMI']

# set up data drift detector
monitor = DataDriftDetector.create_from_datasets(
    ws,
    'mslearn-diabates-drift',
    baseline_data_set,
    target_data_set,
    compute_target=cluster_name, 
    frequency='Week', 
    feature_list=features, 
    drift_threshold=.3, 
    latency=24
)

monitor

#-----BACKFILL_MONITOR---------------------------------------------------------#
'''
Backfill data drift monitor
* backfill the monitor so that it can analyze data drift between the original baseline and the target data
'''
backfill = monitor.backfill(dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(backfill).show()

backfill.wait_for_completion()

#-----ANALYSE_DRIFT------------------------------------------------------------#
# Examine data drift
drift_metrics = backfill.get_metrics()
for metric in drift_metrics:
    print(metric, drift_metrics[metric])