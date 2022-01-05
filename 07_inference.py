# Import libraries
from azureml.core import Environment, Model, Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
import json
import requests

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----MODEL--------------------------------------------------------------------#
# List registered models
for model in Model.list(ws):                                    # Retrieve a list of all models associated with the provided workspace, with optional filters 
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')

# Get the model - by default if model name is specified, latest version will be returned 
model = ws.models['diabetes_model']                             # Get model by name from current workspace
print(model.name, 'version', model.version)

#-----DEPLOY_SERVICE-----------------------------------------------------------#
'''
Deyploy web service
* Web service will be hosted in a container
* Container will need to install any required Python dependencies when it gets initialized
* In this repo, inference code requires scikit-learn and some Azure ML specific packages
* Deploy steps:
    1. Create an environment that includes required dependencies
    2. Add environment to an inference configuration along with the scoring script
    3. Define a deployment configuration for the container in which the environment and script will be hosted
'''
service_env = Environment(name='service-env')                   # Configures a reproducible Python environment for machine learning experiments
python_packages = [                                             # Define dependencies
    'scikit-learn',
    'azureml-defaults',
    'azure-ml-api-sdk'
]

for package in python_packages:                                 
    service_env.python.conda_dependencies.add_pip_package(package)  # Add a pip package

# Create inference config - review ./service/web_service.py for scroing script setup
inference_config = InferenceConfig(                             # Represents configuration settings for a custom environment used for deployment
    source_directory='./service',                               # Path to folder that contains all files to create the image
    entry_script='web_service.py',                              # Path to local file that contains the code to run for the image
    environment=service_env                                     # Environment object to use for the deployment
)

# Configure the web service container
deployment_config = AciWebservice.deploy_configuration(         # Create a configuration object for deploying an AciWebservice
    cpu_cores=1,                                                # CPU cores to allocate for this Webservice, defaults to 0.1
    memory_gb=1                                                 # Memory (in GB) to allocate for this Webservice, defaults to 0.5
)

# Deploy the model as a service
print('Deploying model...')
service_name = 'diabetes-service'
service = Model.deploy(                                         # Deploy a Webservice from zero or more Model objects
    ws,                                                         # Workspace object to associate the Webservice with
    service_name,                                               # Name to give the deployed service
    [model],                                                    # List of model objects
    inference_config,                                           # InferenceConfig object used to determine required model properties
    deployment_config,                                          # ebserviceDeploymentConfiguration used to configure the webservice
    overwrite=True                                              # Indicates whether to overwrite existing service if service with specified name already exists
)

service.wait_for_deployment()                                   # Automatically poll on the running Webservice deployment
print(service.state)

# Review service status
print(service.get_logs())                                       # Retrieve logs for this Webservice

# Need to make a change and redeploy by deleting unhealthy services:
#service.delete()

# Retrieve names of web services in current workspace
for webservice_name in ws.webservices:
    print(webservice_name)

#-----CONSUME_SERVICE_I--------------------------------------------------------#
'''
Consume deployed service
* via Azure ML SDK to:
    * Connect to the containerized web service
    * Generate predictions from diabetes classification model
'''

# Input as an array of two feature arrays
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array or arrays to a serializable list in a JSON document
input_json = json.dumps({'data': x_new})

# Call the web service, passing the input data
predictions = service.run(input_data = input_json)              # Call this Webservice with the provided input

# Get the predicted classes
predicted_classes = json.loads(predictions)
for i in range(len(x_new)):
    print ('Patient {}'.format(x_new[i]), predicted_classes[i] )

#-----CONSUME_SERVICE_II-------------------------------------------------------#
'''
Consume deployed service
* via endpoint and HTTP request
    * Model in production is likely to be consumed by business applications 
    * --> usually do not use the Azure ML SDK
    * Instead, make HTTP requests to the web service
'''
endpoint = service.scoring_uri                                  # Determine the URL
print(endpoint)

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

#-----DELETE_SERVICE-------------------------------------------------------#
# Delete the service when no longer needed
# service.delete()                                              # Delete this Webservice from its associated workspace
# print ('Service deleted.')