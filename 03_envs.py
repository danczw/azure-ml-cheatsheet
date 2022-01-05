# Import libraries
from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
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
experiment_env = Environment.from_conda_specification(          # Create environment object from an environment specification YAML file
    'ml_sdk_env',                                               # Environment name
    'environment.yml'                                           # Conda environment specification YAML file path
)

# Create a Python environment from an existing Conda environment
# env = Environment.from_existing_conda_environment(            # Create environment object created from a locally existing conda environment
    # name='training_environment',                              # Environment name
    # conda_environment_name='py_env'                           # Name of locally existing conda environment
# )

# Create a Python environment by specifying packages manually
# env = Environment('training_environment')                     # Configures reproducible Python environment for machine learning experiments
# deps = CondaDependencies.create(                              # Initialize new CondaDependencies object, returns CondaDependencies object instance with user specified dependencies
    # conda_packages=['scikit-learn','pandas','numpy'],         # List of to be installed conda packages
    # pip_packages=['azureml-defaults']                         # List of to be installed pip packages
# )
# env.python.conda_dependencies = deps                          # Defines the Python environment and interpreter to use on a target compute for a run - Conda dependencies

# Let Azure ML manage dependencies
experiment_env.python.user_managed_dependencies = False         # Indicates whether Azure Machine Learning reuses an existing Python environment

# Print the environment details
print(experiment_env.name, 'defined.')
print(experiment_env.python.conda_dependencies.serialize_to_string())

# Register the environment
experiment_env.register(workspace=ws)                           # Register the environment object in your workspace

# View registered environments
envs = Environment.list(workspace=ws)                           # Return a dictionary containing environments in the workspace
for env in envs:
    print('Name',env)