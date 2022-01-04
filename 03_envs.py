# Import libraries
from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----ENVIRONMENT--------------------------------------------------------------#
'''
Environment
* Encapsulated by the Environment class
* Can be used to create environments and specify runtime configuration for each experiment
* Default: Azure handles environment creation and package installation through creation of Docker containers
    * Custom container images can be created and registered in a container registry
    * Can be used to override default base images
    * --> Use custom container image by modifying the attributes of the environment's docker property
    * Alternatively, create image on-demand based on the base image and additional settings in a dockerfile
'''
# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification('ml_sdk_env', 'environment.yml')

# Create a Python environment from an existing Conda environment
# env = Environment.from_existing_conda_environment(name='training_environment', conda_environment_name='py_env')

# Create a Python environment by specifying packages manually
# env = Environment('training_environment')
# deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy'], pip_packages=['azureml-defaults'])
# env.python.conda_dependencies = deps

# Let Azure ML manage dependencies
experiment_env.python.user_managed_dependencies = False 

# Print the environment details
print(experiment_env.name, 'defined.')
print(experiment_env.python.conda_dependencies.serialize_to_string())

# Register the environment
experiment_env.register(workspace=ws)

# View registered environments
envs = Environment.list(workspace=ws)
for env in envs:
    print('Name',env)