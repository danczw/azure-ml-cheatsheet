import json
import joblib
import numpy as np
import os

#-----WEB_SERVICE--------------------------------------------------------------#
'''
Inference web service
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
# Initialization function when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'diabetes_model.pkl')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])

    # Get a prediction from the model
    predictions = model.predict(data)

    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
        
    # Return the predictions as JSON
    return json.dumps(predicted_classes)