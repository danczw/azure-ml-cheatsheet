# Import libraries
from azureml.core import Environment, Experiment, Model, ScriptRunConfig, Workspace
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.widgets import RunDetails

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')               # Get specified dataset from list of all datasets in workspace

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')          # Get specified environment object from workspace

#-----SCRIPT_SETUP-------------------------------------------------------------#
'''
Hyperparameter tuning
* Hyperparameter: parameter values that influence training, but can't be determined from the training data itself
* E.g.: for logistic regression model tune regularization rate hyperparameter to counteract bias in the model
* Choice of hyperparameter values can significantly affect the performance of a trained model, or the time taken to train it
* Often try of multiple combinations to find the optimal solution is needed
'''
# Review ./experiments/* which includes example pipeline steps
experiment_folder = './experiments'                                 # Pipeline steps folder

script_config = ScriptRunConfig(                                # Represents configuration information for submitting a training run in Azure Machine Learning
    source_directory=experiment_folder,                         # Local directory containing code files needed for a run
    script='09_parameter_tuning.py',                            # File path relative to the source_directory of the script to be run
    # Add non-hyperparameter arguments - in this case, the training dataset
    arguments = [                                               # Optional command line arguments to pass to the training script
        ['--input-data', diabetes_ds.as_named_input('training_data')]   # Reference to tabular dataset
    ],
    environment=registered_env,                                 # Environment to use for run, if no environment is specified, azureml.core.runconfig.DEFAULT_CPU_IMAGE will be used as the Docker image for the run                      
    compute_target=cluster_name,                                # Compute target where training will happen, can be ComputeTarget object, existing ComputeTarget name, or the string "local" (default)
)

'''
Hyperparameter search space
* Discrete hyperparameters
    * choice:                       Returns a choice among discrete values
    * quniform(low, high, q):       Returns a value like round(uniform(low, high) / q) * q
    * qloguniform(low, high, q):    Returns a value like round(exp(uniform(low, high)) / q) * q
    * qnormal(mu, sigma, q):        Returns a value like round(normal(mu, sigma) / q) * q
    * qlognormal(mu, sigma, q):     Returns a value like round(exp(normal(mu, sigma)) / q) * q
* Continuous hyperparameters
    * uniform(low, high):       Returns a value uniformly distributed between low and high
    * loguniform(low, high):    Returns a value drawn according to exp(uniform(low, high)) so that logarithm of return value is uniformly distributed
    * normal(mu, sigma):        Returns a real value that's normally distributed with mean mu and standard deviation sigma
    * lognormal(mu, sigma):     Returns a value drawn according to exp(normal(mu, sigma)) so that logarithm of return value is normally distributed

Sampling hyperparameter search space
* Random sampling: 
    * Supports discrete and continuous hyperparameters
    * Supports early termination of low-performance runs
    * Hyperparameter values are randomly selected from defined search space
* Grid sampling
    * Supports discrete hyperparameters
    * Supports early termination of low-performance runs
    * Simple grid search over all possible values
    * Only be used with choice hyperparameters
* Bayesian sampling
    * Picks samples based on how previous samples did, so that new samples improve the primary metric
    * Supports choice, uniform, and quniform distributions
'''
# Sample a range of parameter values
params = GridParameterSampling(                                 # Defines grid sampling over a hyperparameter search space
    {
        # Hyperdrive will try 6 combinations, adding these as script arguments
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)
    }
)

# Configure hyperdrive settings
hyperdrive = HyperDriveConfig(                                  # Configuration that defines a HyperDrive run
    run_config=script_config,                                   # An object for setting up configuration for script/notebook runs
    hyperparameter_sampling=params,                             # Hyperparameter sampling space
    policy=None,                                                # The early termination policy to use
    primary_metric_name='AUC',                                  # Name of primary metric reported by the experiment runs
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,             # Either PrimaryMetricGoal.MINIMIZE or PrimaryMetricGoal.MAXIMIZE, parameter determines if the primary metric is to be minimized or maximized when evaluating runs
    max_total_runs=6,                                           # Maximum total number of runs to create
    max_concurrent_runs=2                                       # Maximum number of runs to execute concurrently
)

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'mslearn-diabetes-hyperdrive'
experiment = Experiment(                                        # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)
print('Pipeline submitted for execution.')

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure ML
'''
# Submit an experiment incl config to be submitted and return the active created run
run = experiment.submit(                                        # Run defines the base class for all Azure Machine Learning experiment runs
    config=hyperdrive,                                          # Config to be submitted
)                                   
print('Pipeline submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(pipeline_run).show()

run.wait_for_completion()                              # Wait for the completion of this run, returns the status object after the wait

#-----PERFORMANCE--------------------------------------------------------------#
# Print all child runs, sorted by the primary metric
for child_run in run.get_children_sorted_by_primary_metric():   # Return a list of children sorted by their best primary metric
    print(child_run)

# Get the best run, and its metrics and arguments
best_run = run.get_best_run_by_primary_metric()                 # Find and return the Run instance that corresponds to the best performing run amongst all child runs
best_run_metrics = best_run.get_metrics()                       # Retrieve the metrics logged to the run
script_arguments = best_run.get_details()['runDefinition']['arguments'] # Get definition, status information, current log files, and other details of the run
print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Arguments:',script_arguments)

#-----REGISTER_MODEL-----------------------------------------------------------#
'''
Register run machine learning model
* Outputs of the experiment also include the trained model file
* Register model in your Azure ML workspace
* Allowing to track model versions and retrieve them later
'''
best_run.register_model(                                        # Register a model for operationalization
    model_path='outputs/diabetes_model.pkl',                    # Relative cloud path to model
    model_name='diabetes_model',                                # Name of model
    tags={'Training context':'Hyperdrive'},                     # Dictionary of key value tags to assign to model
    properties={                                                # Dictionary of key value properties to assign to model, properties cannot be changed after model creation
        'AUC':  best_run_metrics['AUC'],
        'Accuracy': best_run_metrics['Accuracy']
    }
)