from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

#-----WORKSPACE----------------------------------------------------------------#
# load workspace from config JSON file
ws = Workspace.from_config()
print(ws.name, "loaded")

#-----COMPUTE_CLUSTER----------------------------------------------------------#
"""
Azure Machine Learning supports a range of compute targets
* can be define in your workpace and use to run experiments
* paying for the resources only when using them
* Cluster names must be globally unique names between 2 to 16 characters in length
* Valid characters are letters, digits, and the - character
"""
cluster_name = "ml-sdk-cc"

# Create new compute cluster if not already existing
try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        # define VM size and max number of nodes for scaling
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)

# Check on the status of the compute
cluster_state = training_cluster.get_status()
print(cluster_state.allocation_state, cluster_state.current_node_count)