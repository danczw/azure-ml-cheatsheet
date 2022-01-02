# Import libraries
import os
import numpy as np
from azureml.core import Model
import joblib

#-----WEB_SERVICE--------------------------------------------------------------#
'''
Interference web service
* Web service to deploy the model
* Service steps:
    1. Load the input data
    2. Get the model from the workspace
    3. generate and return predictions
* Note: called entry script (or a scoring script) which will be deployed to web service

Functions:
* init: 
    * Called when the service is initialized
    * Generally used to load the model
    * Uses the AZUREML_MODEL_DIR env variable to determine folder where model is stored
* run:
    * Called each time a client application submits new data
    * Generally used to inference predictions from the model
'''
def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)

def run(mini_batch):
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read the comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for prediction (model expects multiple items)
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList