# Import libraries
from azureml.core import Environment, Model, Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
import json

#-----WORKSPACE----------------------------------------------------------------#
# load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----MODEL--------------------------------------------------------------------#
# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')

# Get the model - by default if model name is specified, latest version will be returned 
model = ws.models['diabetes_model']
print(model.name, 'version', model.version)

#-----DEPLOY_SERVICE-----------------------------------------------------------#
'''
Deyploy web service
* Web service will be hosted in a container
* Container will need to install any required Python dependencies when it gets initialized
* In this repo, interference code requires scikit-learn and some Azure Machine Learning specific packages

Deplyo steps:
1. Create an environment that included these
2. Add environment to an inference configuration along with the scoring script
3. Define a deployment configuration for the container in which the environment and script will be hosted
'''
service_env = Environment(name='service-env')                   # Initialize env
python_packages = [                                             # Define dependencies
    'scikit-learn',
    'azureml-defaults',
    'azure-ml-api-sdk'
]

for package in python_packages:                                 # Add dependencies to env
    service_env.python.conda_dependencies.add_pip_package(package)

# Create interference config
inference_config = InferenceConfig(
    source_directory='./service',                               # Web service entry script location
    entry_script='web_service.py',                              # Web service entry script name
    environment=service_env                                     # Interference env
)

# Configure the web service container
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model as a service
print('Deploying model...')
service_name = 'diabetes-service'
service = Model.deploy(ws, service_name, [model], inference_config, deployment_config, overwrite=True)
service.wait_for_deployment(True)
print(service.state)

# Review service status
print(service.get_logs())

# Need to make a change and redeploy by deleting unhealthy services:
#service.delete()

# Retrieve names of web services in current workspace
for webservice_name in ws.webservices:
    print(webservice_name)

#-----CONSUME_SERVICE_I--------------------------------------------------------#
'''
* Azure Machine Learning SDK to:
    * Connect to the containerized web service
    * Generate predictions from diabetes classification model
'''

# Input as an array of two feature arrays
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array or arrays to a serializable list in a JSON document
input_json = json.dumps({'data': x_new})

# Call the web service, passing the input data
predictions = service.run(input_data = input_json)

# Get the predicted classes.
predicted_classes = json.loads(predictions)
for i in range(len(x_new)):
    print ('Patient {}'.format(x_new[i]), predicted_classes[i] )

#-----CONSUME_SERVICE_II-------------------------------------------------------#
'''
* In production, a model is likely to be consumed by business applications that do not use the Azure Machine Learning SDK, but simply make HTTP requests to the web service
'''