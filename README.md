# Azure Machine Learning Python SDK Cheat Sheet

This repository is aggregating main functions, methods and ultimately the setup of the Azure Machine Learning Python SDK.

Azure Machine Learning is a cloud-based service for creating and managing machine learning solutions. It's designed to help data scientists and machine learning engineers leverage their existing data processing and model development skills and frameworks, and scale their workloads to the cloud.


Scripts are in order of implementation with ´./04_experiment_setup.py´ aggregating all previous shown functions, methods and setups - *Overview*:
1. datastores: upload and registration of training data [./datastores.py](./datastores.py)
2. environment: setup of experiment environment and dependencies [./envs.py](./envs.py) 
3. compute target: creating compute target for model training ([./03_compute.py](./03_compute.py))
4. setup: running experiments training models from [./experiments](./experiments) 

Few notes on the SDK:
* azureml-core: Azure Machine Learning Python SDK enabling to write code that uses resources in your Azure Machine Learning workspace
* azureml-mlflow: open source platform for managing machine learning processes which can also be used to track metrics as an alternative to the native log functionality (not used in this repo)