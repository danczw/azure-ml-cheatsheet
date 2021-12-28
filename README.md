# Azure Machine Learning Python SDK Cheat Sheet

This repository is aggregating main functions, methods and ultimately the setup of the Azure Machine Learning Python SDK.

Azure Machine Learning is a cloud-based service for creating and managing machine learning solutions. It's designed to help data scientists and machine learning engineers leverage their existing data processing and model development skills and frameworks, and scale their workloads to the cloud.

## Overview
Scripts are in order of implementation with [./04_setup.py](./04_setup.py) aggregating all previous shown functions, methods and setups.
1. datastores: upload and registration of training data - [./01_datastores.py](./01_datastores.py)
2. environment: setup of experiment environment and dependencies - [./02_envs.py](./02_envs.py)
3. compute target: creating compute target for model training - [./03_compute.py](./03_compute.py)
4. setup: running experiments training models passed from any [./experiments](./experiments) - [./04_setup.py](./04_setup.py)

## Setup
* Setup Azure ML workspace ([MS ML Quickstart](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources))
* Update [./config_template.json](./config_template.json) with your Azure subscription id, resource group and workspace name - **ATTENTION** make sure rename file to `config.json`
* run `conda env create -f environment.yml` for Python environment for local development

Few notes on the SDK libraries:
* azureml-core: Azure Machine Learning Python SDK enabling to write code that uses resources in your Azure Machine Learning workspace
* azureml-mlflow: open source platform for managing machine learning processes which can also be used to track metrics as an alternative to the native log functionality (not used in this repo)