# Import libraries
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, 'loaded')

#-----COMPUTE_CLUSTER----------------------------------------------------------#
'''
Azure Machine Learning supports a range of compute targets
* Can be define in your workpace and be used to run experiments
* Types:
    * Local compute:
        * Runs experiment on same compute target as code used to initiate experiment
        * e.g. physical workstation or VM
    * Compute cluster
        * Azure Machine Learning compute clusters
        * Multi-node clusters of Virtual Machines
        * Can automatically scale up or down to meet demand
        * See table at end for available Azure managed compute targets / clusters
    * Attached compute
        * Attached compute instances to the Azure ML workspace
        * e.g. Virtual Machine or Azure Databricks cluster
'''
# Create new compute cluster if not already existing
cluster_name = "ml-sdk-cc"
try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        # Define VM size and max number of nodes for scaling
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)

# Check on the status of the compute
cluster_state = training_cluster.get_status()
print(cluster_state.allocation_state, cluster_state.current_node_count)

'''
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
'''