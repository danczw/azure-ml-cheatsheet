# Import libraries
import argparse
from azureml.core import Run
import pandas as pd
import os

# Get the experiment run context
run = Run.get_context() 	                                        # method to retrieve the experiment run context when the script is run

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

args = parser.parse_args()                                      # Add arguments to args collection

#-----DATA---------------------------------------------------------------------#
# load the data (passed as an input dataset)
print('Loading Data...')
data = run.input_datasets['raw_data'].to_pandas_dataframe()

#-----PROCESSING_&_LOGGING-----------------------------------------------------#
# Count the rows and log the result
row_count = (len(data))
run.log('observations', row_count)
print('Analyzing {} rows of data'.format(row_count))

# Count and log the label counts
diabetic_counts = data['Diabetic'].value_counts()
print(diabetic_counts)
for k, v in diabetic_counts.items():
    run.log('Label:' + str(k), v)

#-----SAVE---------------------------------------------------------------------#
# Save a sample of the data in the outputs folder (which gets uploaded automatically)
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()