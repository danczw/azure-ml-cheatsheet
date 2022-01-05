# Import libraries
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

#-----WORKSPACE----------------------------------------------------------------#
# Load workspace from config JSON file
ws = Workspace.from_config()                                    # Returns a workspace object based on config file 
print(ws.name, 'loaded')

#-----COMPUTE_CLUSTER----------------------------------------------------------#
'''
Azure ML supports a range of compute targets
* Can be define in your workpace and be used to run experiments
* Types:
    * Local compute:
        * Runs experiment on same compute target as code used to initiate experiment
        * e.g. physical workstation or VM
    * Compute cluster
        * Azure ML compute clusters
        * Multi-node clusters of Virtual Machines
        * Can automatically scale up or down to meet demand
        * See table at end for available Azure managed compute targets / clusters
    * Attached compute
        * Attached compute instances to the Azure ML workspace
        * e.g. Virtual Machine or Azure Databricks cluster

Note: view different compute types in the README.md
'''

# View compute resources in workspace
for compute_name in ws.compute_targets:                         # List all compute targets in workspace, returns dictionary with key as compute target name and value as ComputeTarget object
    compute = ws.compute_targets[compute_name]
    print('\t', compute.name, ':', compute.type)

# Create new compute cluster if not already existing
cluster_name = "ml-sdk-cc"
try:
    # Check for existing compute target
    training_cluster = ComputeTarget(                           # Abstract parent class for all compute targets managed by Azure ML
        workspace=ws,                                           # Workspace object containing the Compute object to retrieve
        name=cluster_name                                       # Name of the Compute object to retrieve
    )
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        # Define VM size and max number of nodes for scaling
        compute_config = AmlCompute.provisioning_configuration( # Create a configuration object for provisioning an AmlCompute target
            vm_size='STANDARD_DS11_V2',                         # Size of agent VMs
            max_nodes=2                                         # Maximum number of nodes to use on the cluster, defaults to 4
        )
        training_cluster = ComputeTarget.create(                # Provision new Compute object by specifying a compute type and related configuration               
            ws,                                                 # Workspace object to create the Compute object under
            cluster_name,                                       # Name to associate with the Compute object
            compute_config                                      # ComputeTargetProvisioningConfiguration object used to determine type of Compute object to provision
        )
        training_cluster.wait_for_completion(show_output=True)  # Wait for the current provisioning operation to finish on the cluster
    except Exception as ex:
        print(ex)

# Check on the status of the compute
cluster_state = training_cluster.get_status()                   # Retrieve the current provisioning state of the Compute object
print(cluster_state.allocation_state, cluster_state.current_node_count)