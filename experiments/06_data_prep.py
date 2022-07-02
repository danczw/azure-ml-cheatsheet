# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

# Method to retrieve the experiment run context when the script is run
run = Run.get_context()

#-----EXPERIMENT_PARAMETER-----------------------------------------------------#
'''
Experiment Parameter
* Increase the flexibility of your training experiment by adding parameters to your script
* Enabling to repeat the same training experiment with different settings
'''
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-data',
    type=str,
    dest='raw_dataset_id',
    help='raw dataset'
)
# parser.add_argument('--input-data', type=str, dest='dataset_folder', help='data mount point') # Using file dataset

parser.add_argument(
    '--prepped-data',
    type=str,
    dest='prepped_data',
    default='prepped_data',
    help='Folder for results'
)

# Add arguments to args collection
args = parser.parse_args()
save_folder = args.prepped_data

#-----DATA---------------------------------------------------------------------#
# load the data (passed as an input dataset)
print('Loading Data...')
diabetes = run.input_datasets['raw_data'].to_pandas_dataframe()

# Using file dataset instead of tabular data:
    # data_path = run.input_datasets['training_files'] # Get the training data path from the input using a file dataset
    # (You could also just use args.dataset_folder if you don't want to rely on a hard-coded friendly name)
    # all_files = glob.glob(data_path + '/*.csv') # Read the files
    # diabetes = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

# Log raw row count
row_count = (len(diabetes))
run.log('raw_rows', row_count)

#-----DATA_PREP----------------------------------------------------------------#
# remove nulls
diabetes = diabetes.dropna()

# Normalize the numeric columns
scaler = MinMaxScaler()
num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
diabetes[num_cols] = scaler.fit_transform(diabetes[num_cols])

# Log processed rows
row_count = (len(diabetes))
run.log('processed_rows', row_count)

#-----SAVE---------------------------------------------------------------------#
# Save the prepped data
print('Saving Data...')
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
diabetes.to_csv(save_path, index=False, header=True)

# End the run
run.complete()