from azureml.core import Environment, Workspace

#-----WORKSPACE----------------------------------------------------------------#
# load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, "loaded")

#-----ENVIRONMENT--------------------------------------------------------------#
# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification("experiment_env", "environment.yml")

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
    print("Name",env)