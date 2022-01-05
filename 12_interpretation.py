# Import libraries
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration
from azureml.interpret import ExplanationClient

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----DATASET------------------------------------------------------------------#
# Get the training dataset from registered datasets (see ./01_datastores.py)
diabetes_ds = ws.datasets.get('diabetes dataset')               # Get specified dataset from list of all datasets in workspace

# Split for train and test dataset
train_ds, test_ds = diabetes_ds.random_split(percentage=0.7, seed=123)

#-----COMPUTE_TARGET-----------------------------------------------------------#
# Define compute target (see ./02_compute.py)
cluster_name = 'ml-sdk-cc'

#-----ENVIRONMENT_SETUP--------------------------------------------------------#
# Get the registered environment (see ./03_envs.py)
registered_env = Environment.get(ws, 'experiment_env')          # Get specified environment object from workspace

#-----SCRIPT_SETUP-------------------------------------------------------------#
'''
Experiment script
* Experiment is a named process, usually the running of a single script or a pipeline
* Can generate metrics and outputs and be tracked in the Azure ML workspace
* Experiment can be run multiple times, with different data, code, or settings
* Azure ML tracks each run, enabling to view run history and compare results for each run
* Single step experiments
    * For running single step experiments, no pipeline need to be deployed
    * Create a script run config - similar to a single pipeline step (see 05_pipeline.py)
        * Identifies the Python script file to be run in the experiment
        * Determines the compute target and Python environment
        * Creates a DockerConfiguration for the script run
'''

# Review ./experiments/* which includes interpret model file
experiment_folder = './experiments'

script_config = ScriptRunConfig(                                # Represents configuration information for submitting a training run in Azure Machine Learning
    source_directory=experiment_folder,                         # Local directory containing code files needed for a run
    script='12_interpret_model.py',                             # File path relative to the source_directory of the script to be run
    arguments = [                                               # Optional command line arguments to pass to the training script
        '--input-data', diabetes_ds.as_named_input('raw_data')  # Reference to tabular dataset
    ],
    environment=registered_env,                                 # Environment to use for run, if no environment is specified, azureml.core.runconfig.DEFAULT_CPU_IMAGE will be used as the Docker image for the run                      
    compute_target=cluster_name,                                # Compute target where training will happen, can be ComputeTarget object, existing ComputeTarget name, or the string "local" (default)
    docker_runtime_config=DockerConfiguration(use_docker=True)  # For jobs that require Docker runtime-specific configurations
)

#-----EXPERIMENT---------------------------------------------------------------#
# Create an Azure ML experiment in workspace
experiment_name = 'ml-sdk-interpret'
experiment = Experiment(                                        # Main entry point class for creating and working with experiments in Azure Machine Learning
    workspace=ws,                                               # Workspace object containing the experiment
    name=experiment_name                                        # Experiment name
)

#-----RUN----------------------------------------------------------------------#
'''
Run object is a reference to an individual run of an experiment in Azure ML
'''
# Submit an experiment incl config to be submitted and return the active created run
run = experiment.submit(config=script_config)                   # Run defines the base class for all Azure Machine Learning experiment runs                
print('Experiment submitted for execution.')

# In Jupyter Notebooks, use RunDetails widget to see a visualization of the run details
# RunDetails(run).show()

run.wait_for_completion()                                       # Wait for the completion of this run, returns the status object after the wait

#-----EXPLANATION--------------------------------------------------------------#
'''
Feature Importance 
* Creating explainers - there are multiple types of explainer, including:
	* MimicExplainer:
        * Creates a global surrogate model that approximates your trained model and can be used to generate explanations
        * Explainable model must have the same kind of architecture as your trained model (for example, linear or tree-based)
    * TabularExplainer:
        * Acts as wrapper around various SHAP explainer algorithms
        * Automatically choosing the one that is most appropriate for model architecture
    * PFIExplainer:
        * Permutation Feature Importance explainer
        * Analyzing feature importance by shuffling feature values and measuring the impact on prediction performance.
* To retrieve global importance values for the features call the explain_global() method of your explainer to get a global explanation
* Use get_feature_importance_dict() method to get a dictionary of the feature importance values

Feature Importance
* Global feature importance
    * Quantifies the relative importance of each feature in the test dataset as a whole
	* Provides a general comparison of the extent to which each feature in the dataset influences prediction
* Local feature importance
    * Measures influence of each feature value for a specific individual prediction
	* Multi-class classification model:
        * Local importance values for each possible class is calculated for every feature
        * Total importance value across all classes of this prediction always being 0
	* Regression model:
        * Local importance values simply indicate the level of influence each feature has on the predicted scalar label
'''
# Get the feature explanations
client = ExplanationClient.from_run(run)                        # Defines client that uploads and downloads explanations: create client with factory method given a run
engineered_explanations = client.download_model_explanation()   # Download a model explanation that has been stored in run history
feature_importances = engineered_explanations.get_feature_importance_dict()     # Dict with global feature importance

# Overall feature importance
print('Feature\tImportance')
for key, value in feature_importances.items():
    print(key, '\t', value)

# TODO: add local feature importance
