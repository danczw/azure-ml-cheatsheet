# Azure Machine Learning Python SDK Cheat Sheet

This repository is aggregating main functions, methods and ultimately the setup of the Azure Machine Learning Python SDK.

Azure Machine Learning is a cloud-based service for creating and managing machine learning solutions. It's designed to help data scientists and machine learning engineers leverage their existing data processing and model development skills and frameworks, and scale their workloads to the cloud.

------------------------------------------------
------------------------------------------------

## Overview
Scripts are in order of implementation:
1. Datastores: upload and registration of training data - [01_datastores.py](./01_datastores.py)
2. Compute: creating compute target for model training - [02_compute.py](./02_compute.py)
3. Environment: setup of experiment environment and dependencies - [03_envs.py](./03_envs.py)
4. Experiment: run single script experiment - [04_experiment.py](./04_experiment.py)
6. Pipeline: running experiments divided into pipeline steps - [06_pipeline.py](./06_pipeline.py)
7. Inference: set up web service and consume for inference - [07_inference.py](./07_inference.py)
8. Batch pipeline and inference: setup pipeline and inference for batch processing - [08_batch_pipeline_inference.py](./08_batch_pipeline_inference.py)
9. Hyperparamenter: tune and evaluate hyperparameter for model performance - [09_hyperparameter.py](./09_hyperparameter.py)
10. Automated ML: run automated Machine Learning experiments for various model testing - [10_automatedML.py](./10_automatedML.py)
11. Privacy: explore differential privacy 
12. Interpretation: explain global and local feature importance - [12_interpretation.py](./12_interpretation.py)

------------------------------------------------
------------------------------------------------

## Setup
- Setup Azure ML workspace ([MS ML Quickstart](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources))
- Update [./config_template.json](./config_template.json) with your Azure subscription id, resource group and workspace name - **ATTENTION!** make sure rename file to `config.json`
- Update `template.env`
- Run `conda env create -f environment.yml` for Python environment for local development
- Pipeline steps ("experiments") can be found in [./experiments](./experiments) as detailed below

------------------------------------------------
------------------------------------------------

## Further Notes

The basic single script experiment of 04. includes some basic data processing and logging as per [./experiments/04_experiment_script.py](./experiments/04_experiment_script.py)

For MLFlow integration in 05., the single script has been extended as per [./experiments/04_experiment_script_mlflow.py](./experiments/04_experiment_script_mlflow.py)

The main pipeline of 06. itself includes two steps:
- Data preprocessing [./experiments/06_data_prep.py](./experiments/06_data_prep.py) inkl. normalization
- Model training [./experiments/06_train_model.py](./experiments/06_train_model.py) for classification using logistic regression

Finalized Azure Machine Learning pipeline ([06_pipeline.py](./06_pipeline.py)) will look like:

![Azure ML Pipeline with two steps](./assets/pipeline_run.png "Azure ML Pipeline")

For hyperparameter tuning of 09., the model training has been adepted as per [./experiments/09_parameter_tuning.py](./experiments/09_parameter_tuning.py).

Interpretation of model as in part 12. has been included in the experiment as per [./experiments/12_interpret_model.py](./experiments/12_interpret_model.py) 

------------------------------------------------

Azure Compute Targets:
#==============================================================================#
| Compute target         | Usage       | Description                           |
|==============================================================================| 
| Local web service      | Testing /   | Good for limited testing and          |
|                        | debug       | troubleshooting.                      |
|------------------------------------------------------------------------------| 
| Azure Kubernetes       | Real-time   | Good for high-scale production        |
| Service (AKS)	         | inference   | deployments. Provides autoscaling,    |
|                        |             | and fast response times.              |
|------------------------------------------------------------------------------|
| Azure Container        | Testing     | Good for low scale, CPU-based         |
| Instances (ACI)        |             | workloads.                            |
|------------------------------------------------------------------------------|
| Azure Machine Learning | Batch       | Run batch scoring on serverless       |
| Compute Clusters	     | inference   | compute. Supports normal and          |
|                        |             | low-priority VMs.                     |
|------------------------------------------------------------------------------|
| Azure IoT Edge         | IoT         | Deploy & serve ML models on           |
| (Preview)              | module      | IoT devices.                          |
#==============================================================================#

------------------------------------------------

Few notes on the SDK libraries:
- azureml-automl:
- azureml-core: Azure Machine Learning Python SDK enabling to write code that uses resources in your Azure Machine Learning workspace
- azureml-explain-model:
- azureml-mlflow: open source platform for managing machine learning processes which can also be used to track metrics as an alternative to the native log functionality
- azureml-pipeline:

------------------------------------------------
------------------------------------------------

## TODO
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