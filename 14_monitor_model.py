# Import libraries
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
import json
import requests

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----WEB_SERVICE--------------------------------------------------------------#
# Retrieve names of web services in current workspace
for webservice_name in ws.webservices:
    print(webservice_name)

service = Webservice(ws, 'diabetes-service-app-insights')

#-----APPINSIGHTS--------------------------------------------------------------#
'''
Application Insights
* Application performance management service
* Enables the capture, storage, and analysis of telemetry data from applications
* Necessary package is already included in Azure Machine Learning Web services
'''
# Enable AppInsights
service.update(enable_app_insights=True)
print('AppInsights enabled!')

#-----SERVICE------------------------------------------------------------------#
endpoint = service.scoring_uri                                  # Determine the URL
print(endpoint)

#-----CONSUME_SERVICE----------------------------------------------------------#
'''
Consume deployed service
* Via endpoint and HTTP request
    * Model in production is likely to be consumed by business applications 
    * --> usually do not use the Azure Machine Learning SDK
    * Instead, make HTTP requests to the web service
'''

# Input as an array of two feature arrays
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Set the content type
headers = { 'Content-Type':'application/json' }

# Call the web service, passing the input data
predictions = requests.post(endpoint, input_json, headers = headers)

# Get the predicted classes
predicted_classes = json.loads(predictions.json())
for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )

# Now you can view the data logged for the service endpoint in the Azure portal
