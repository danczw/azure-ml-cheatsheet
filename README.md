# Azure Machine Learning Python SDK Cheat Sheet

This repository is aggregating main functions, methods and ultimately the setup of the Azure Machine Learning Python SDK.

Azure Machine Learning is a cloud-based service for creating and managing machine learning solutions. It's designed to help data scientists and machine learning engineers leverage their existing data processing and model development skills and frameworks, and scale their workloads to the cloud.

Goal is to work / read through the scripts based on the overview order to get a better understanding of Azure ML Python SDK with as limited docs review needed as possible. 

<img src="./assets/azure_ml.jpg" alt="Azure ML" width="400"/>

<br>

Overview
=====

Scripts are in order of implementation:

1. __Datastores__
    - Upload and registration of training data
    - Main script: [01_datastores.py](./01_datastores.py)
2. __Compute__
    - Creating compute target for model training
    - Main script: [02_compute.py](./02_compute.py)
3. __Environment__
    - Setup of experiment environment and dependencies
    - Main script: [03_envs.py](./03_envs.py)
4. __Experiment__
    - Run single script experiment
    - Main script: [04_experiment.py](./04_experiment.py)
5. __ML Flow__
    - Manage the Machine Learning development lifecycle
    - Main script: [05_mlflow.py](./05_mlflow.py)
6. __Pipeline__
    - Running experiments divided into pipeline steps
    - Main script: [06_pipeline.py](./06_pipeline.py)
7. __Inference__
    - Set up web service and consume for inference
    - Main script: [07_inference.py](./07_inference.py)
8. __Batch pipeline and inference__
    - Setup pipeline and inference for batch processing
    - Main script: [08_batch_pipeline_inference.py](./08_batch_pipeline_inference.py)
9. __Hyperparamenter__
    - Tune and evaluate hyperparameter for model performance
    - Main script: [09_hyperparameter.py](./09_hyperparameter.py)
10. __Automated ML__
    - Run automated Machine Learning experiments for various model testing
    - Main script: [10_automatedML.py](./10_automatedML.py)
11. __Privacy__
    - Explore differential privacy to preserve the privacy of individual data points
    - Main script: [11_privacy.py](./11_privacy.py)
12. __Interpretation__
    - Explain global and local feature importance
    - Main script: [12_interpretation.py](./12_interpretation.py)
13. __Unfairness__
    - Analyze disparity between prediction performance across sensitive feature groups
    - Main script: [13_unfairness.py](./13_unfairness.py)
14. __Monitor model__
    - Monitor telemetry from models on Azure ML
    - Main script: [14_monitor_model.py](./14_monitor_model.py)
15. __Data drift__
    - Capture new data in a dataset and compare to dataset with which model is trained
    - Main script: [15_data_drift.py](./15_data_drift.py)

<br>

Setup
=====

- Setup Azure ML workspace ([MS ML Quickstart](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources))
- Update [./config_template.json](./config_template.json) with your Azure subscription id, resource group and workspace name
    - **ATTENTION!** make sure to rename file to `config.json`
- Update [./template.env](./template.env)
    - **ATTENTION!** make sure to rename file to `.env`
- Run `conda env create -f environment.yml` for Python environment for local development
- Pipeline steps ("experiments") can be found in [./experiments](./experiments) as detailed below

<br>

Further Notes
=====

### A) Experiment / Pipeline Steps

* The basic single script experiment of 04. includes some basic data processing and logging as per [./experiments/04_experiment_script.py](./experiments/04_experiment_script.py)
* For MLFlow integration in 05., the single script has been extended as per [./experiments/04_experiment_script_mlflow.py](./experiments/04_experiment_script_mlflow.py)
* The main pipeline of 06. itself includes two steps:
    1. Data preprocessing [./experiments/06_data_prep.py](./experiments/06_data_prep.py) inkl. normalization
    2. Model training [./experiments/06_train_model.py](./experiments/06_train_model.py) for classification using logistic regression
* Finalized Azure Machine Learning pipeline ([06_pipeline.py](./06_pipeline.py)) will look like:

<img src="./assets/pipeline_run.png" alt="Azure ML Pipeline" width="300"/>

* For hyperparameter tuning of 09., the model training has been adepted as per [./experiments/09_parameter_tuning.py](./experiments/09_parameter_tuning.py).
* Interpretation of model as in part 12. has been included in the experiment as per [./experiments/12_interpret_model.py](./experiments/12_interpret_model.py) 

<br>

### B) Azure Compute Targets

<table>
  <tr>
    <th>Compute target</th>
    <th>Usage</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Local web service</td>
    <td>Testing / debug</td>
    <td>Good for limited testing and troubleshooting.</td>
  </tr>
  <tr>
    <td>Azure Kubernetes Service (AKS)</td>
    <td>Real-time inference</td>
    <td>Good for high-scale production deployments. Provides autoscaling, and fast response times.</td>
  </tr>
  <tr>
    <td>Azure Container Instances (ACI)</td>
    <td>Testing</td>
    <td>Good for low scale, CPU-based workloads.</td>
  </tr>
  <tr>
    <td>Azure Machine Learning Compute Clusters</td>
    <td>Batch inference</td>
    <td>Run batch scoring on serverless compute. Supports normal and low-priority VMs.</td>
  </tr>
  <tr>
    <td>Azure IoT Edge (Preview)</td>
    <td>IoT module</td>
    <td>Deploy & serve ML models on IoT devices.</td>
  </tr>
</table>

The compute in Azure ML looks like:

<img src="./assets/compute_contexts.png" alt="Azure ML Compute Context" width="300"/>

<br>

### C) Azure ML Python SDK libraries

Overview of main Azure ML SDK libraries used in this repo.

- __azureml-automl-core__
    - Contains automated machine learning classes for executing runs in Azure Machine Learning.
    - Packages: `core`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-automl-core/?view=azure-ml-py)
- __azureml-automl-runtime__
    - Contains automated machine learning classes for executing runs and transforming data in Azure Machine Learning.
    - Packages: `runtimer`, `transformer`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/?view=azure-ml-py)
- __azureml-core__
    - Contains core packages, modules, and classes for Azure Machine Learning: managing compute targets, creating/managing workspaces and experiments, and submitting/accessing model runs and run output/logging.
    - Contains modules supporting data representation for Datastore and Dataset in Azure Machine Learning.
    - Contains Azure Machine Learning exception classes.
    - Packages: `core`, `data`, `exceptions`, `history`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-core/?view=azure-ml-py)
- __azureml-datadrift__
    - Contains functionality to detect when model training data has drifted from its scoring data.
    - Packages: `datadrift`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-datadrift/?view=azure-ml-py)
- __azureml-interpret__
    - Contains functionality for working with model interpretability in Azure Machine Learning.
    - Packages: `interpret`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-interpret/?view=azure-ml-py)
- __azureml-mlflow__
    - Contains functionality integrating Azure Machine Learning with MLFlow: open source platform for managing machine learning processes which can also be used to track metrics as an alternative to the native log functionality
    - Packages: `mlflow`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-mlflow/?view=azure-ml-py)
- __azureml-pipeline-core__
    - Contains core functionality for Azure Machine Learning pipelines, which are configurable machine learning workflows.
    - Packages: `core`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/?view=azure-ml-py)
- __azureml-pipeline-steps__
    - Contains pre-built steps that can be executed in an Azure Machine Learning Pipeline.
    - Packages: `steps`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/?view=azure-ml-py)
- __azureml-train-runtime__
    - Package containing modules providing resources for configuring, managing pipelines, and examining run output for automated machine learning experiments.
    - Packages: `automl`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-runtime/?view=azure-ml-py)
- __azureml-train-core__
    - Contains estimators used in Deep Neural Network (DNN) training.
    - Contains base estimator classes and the generic estimator class in Azure Machine Learning.
    - Contains modules and classes supporting hyperparameter tuning.
    - Contains an estimator for training with Scikit-Learn.
    - Packages: `dnn`, `estimator`, `hyperdrive`
    - Modules: `sklearn`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-train-core/?view=azure-ml-py)
- __azureml-widgets__
    - Contains functionality to view the progress of machine learning training runs in Jupyter Notebooks.
    - Packages: `widgets`
    - [Microsoft Docs](https://docs.microsoft.com/en-us/python/api/azureml-widgets/?view=azure-ml-py)

<br>

TODOs
=====
- [x] start with Workspace definiton
- [x] add 'datastore' registration
- [x] add 'compute target' creation
- [x] add 'environment' definition
- [x] add 'pipeline' configuration
- [x] add 'inference' functionality
- [x] add batch processing
- [x] add 'hyperparameter' tuning 
- [x] add 'automated ml' functionality
- [] review and extend azureml function parameter comments
- [x] add part 'MLflow'
- [x] add part 'differential privacy'
- [x] add part 'interpret models'
- [] add part 'monitor model'
- [] add part 'detect unfairness'
- [] add part 'data drift'