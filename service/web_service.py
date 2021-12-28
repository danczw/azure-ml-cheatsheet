import json
import joblib
import numpy as np
import os

#-----WEB_SERVICE--------------------------------------------------------------#
'''
Interference web service
* web service to deploy the model
    * load the input data
    * get the model from the workspace
    * generate and return predictions
* --> called entry script (or a scoring script) that will be deployed to the web service

Functions:
* init: 
    * called when the service is initialized
    * generally used to load the model
    * uses the AZUREML_MODEL_DIR environment variable to determine the folder where the model is stored
* run:
    * called each time a client application submits new data
    * generally used to inference predictions from the model
'''
# Called when the service is loaded
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