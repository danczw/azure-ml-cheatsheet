# Import libraries
from azureml.core import Environment, Experiment, Model, ScriptRunConfig, Workspace
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.widgets import RunDetails

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')

#-----SCRIPT_SETUP-------------------------------------------------------------#
'''
Hyperparameter tuning
* hyperparameter: parameter values that influence training, but can't be determined from the training data itself
* e.g.: for logistic regression model tune regularization rate hyperparameter to counteract bias in the model
* choice of hyperparameter values can significantly affect the performance of a trained model, or the time taken to train it
* often try of multiple combinations to find the optimal solution is needed
'''
# Review ./experiments/* which includes example pipeline steps
experiment_folder = './experiments' # Pipeline steps folder

# Create a script config
script_config = ScriptRunConfig(
    source_directory=experiment_folder,                         # Step py file location
    script='parameter_tuning.py',                               # Step py file name
    # Add non-hyperparameter arguments -in this case, the training dataset
    arguments = ['--input-data', diabetes_ds.as_named_input('training_data')],  # Reference to input dataset
    environment=registered_env,                                 # Experiment env
    compute_target = cluster_name                               # Compute target
)

# Sample a range of parameter values
params = GridParameterSampling(
    {
        # Hyperdrive will try 6 combinations, adding these as script arguments
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)
    }
)

# Configure hyperdrive settings
hyperdrive = HyperDriveConfig(
    run_config=script_config, 
    hyperparameter_sampling=params, 
    policy=None,                                                # No early stopping policy
    primary_metric_name='AUC',                                  # Find the highest AUC metric
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
    max_total_runs=6,                                           # Restict the experiment to 6 iterations
    max_concurrent_runs=2                                       # Run up to 2 iterations in parallel
)

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'mslearn-diabetes-hyperdrive'
experiment = Experiment(workspace=ws, name='mslearn-diabetes-hyperdrive')
print('Pipeline submitted for execution.')

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure Machine Learning
'''
run = experiment.submit(config=hyperdrive)

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(pipeline_run).show()

run.wait_for_completion(show_output=True)

#-----PERFORMANCE--------------------------------------------------------------#
# Print all child runs, sorted by the primary metric
for child_run in run.get_children_sorted_by_primary_metric():
    print(child_run)

# Get the best run, and its metrics and arguments
best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
script_arguments = best_run.get_details() ['runDefinition']['arguments']
print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Arguments:',script_arguments)

#-----REGISTER_MODEL-----------------------------------------------------------#
'''
Register run machine learning model
* Outputs of the experiment also include the trained model file
* Register model in your Azure Machine Learning workspace
* Allowing to track model versions and retrieve them later
'''
best_run.register_model(
    model_path='outputs/diabetes_model.pkl',
    model_name='diabetes_model',
    tags={'Training context':'Hyperdrive'},
    properties={'AUC': best_run_metrics['AUC'], 'Accuracy': best_run_metrics['Accuracy']}
)

# List registered models
# for model in Model.list(ws):
#     print(model.name, 'version:', model.version)
#     for tag_name in model.tags:
#         tag = model.tags[tag_name]
#         print ('\t',tag_name, ':', tag)
#     for prop_name in model.properties:
#         prop = model.properties[prop_name]
#         print ('\t',prop_name, ':', prop)
#     print('\n')
