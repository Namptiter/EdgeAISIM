from edge_sim_py import *
import math
import os
import random
import msgpack
import pandas as pd
import matplotlib.pyplot as plt


edge_rewards = {}  # Initialise overall dict for rewards of migrating to a edge server
edge_counts = {}  # Inititalise the count dict to count the number of times the service has been migrated to a edge server
edge_ucb_scores = {}  # Initialise the upper confidence bound score dict


# Custom collect method to measure the power consumption of each server
def custom_collect_method(self) -> dict:
    metrics = {
        "Instance ID": self.id,
        "Power Consumption": self.get_power_consumption(),
    }
    return metrics


# List to store total power consumption everytime the task scheduling algorithm is used
power_list = list()


def my_algorithm(parameters):

    print("\n\n")
    total_power = 0  # We sum the power consumption after migrating each service

    for service in Service.all():  # Iterate over every service

        if not service.being_provisioned:  # If service needs to be migrated

            for edge_server in EdgeServer.all():  # iterate through every edge server

                if edge_counts[service][edge_server] == 0:
                    # If an edge server has not been explored yet, assign a high UCB score
                    ucb_score = float('inf')
                else:
                    # Calculate the UCB score using the Upper Confidence Bound formula
                    exploration_term = math.sqrt(math.log(
                        sum(edge_counts[service].values())) / edge_counts[service][edge_server])
                    ucb_score = edge_rewards[service][edge_server] / \
                        edge_counts[service][edge_server] + \
                        0.5 * exploration_term

                edge_ucb_scores[service][edge_server] = ucb_score
            # selected_edge_server = max(edge_ucb_scores[service], key=edge_ucb_scores[service].get)

            # sort every edge server by upper confidence bound score
            edge_servers = sorted(
                EdgeServer.all(),
                key=lambda s: edge_ucb_scores[service][s],
                reverse=True,
            )

            for selected_edge_server in edge_servers:  # iterate through sorted list

                if selected_edge_server.has_capacity_to_host(service=service):
                    # If server can host our service

                    # To save resource, we don't migrate it back to the current edge server
                    if service.server != selected_edge_server:
                        print(
                            f"[STEP {parameters['current_step']}] Migrating {service} From {service.server} to {selected_edge_server}")

                        # Migrate to edge server and increment count
                        service.provision(target_server=selected_edge_server)
                        edge_counts[service][selected_edge_server] += 1

                        # get the sum of powerconsumption of each edge server
                        power = 0
                        for iter_edge_server in EdgeServer.all():
                            power = power + iter_edge_server.get_power_consumption()
                        # After start migrating the service we can move on to the next service
                        total_power += power
                        break

        for edge_server in EdgeServer.all():
            # print(edge_server.cpu,edge_server.memory,edge_server.disk)
            edge_rewards[service][edge_server] += 1 / \
                edge_server.get_power_consumption()

    # Append to power_list for plotting
    power_list.append(total_power)


def stopping_criterion(model: object):
    # As EdgeSimPy will halt the simulation whenever this function returns True,
    # its output will be a boolean expression that checks if the current time step is 600
    return model.schedule.steps == 1000


simulator = Simulator(
    dump_interval=5,
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
)

# Loading a sample dataset
simulator.initialize(input_file="sample_dataset3.json")


# Assigning the custom collect method
EdgeServer.collect = custom_collect_method

# Initialise dicts for each service
for service in Service.all():
    if service not in edge_rewards:
        edge_rewards[service] = {}

    if service not in edge_counts:
        edge_counts[service] = {}

    if service not in edge_ucb_scores:
        edge_ucb_scores[service] = {}

    for edge_server in EdgeServer.all():
        if edge_server not in edge_rewards:
            # Initialize rewards and counts for each service, and for each service, eachedge server
            edge_rewards[service][edge_server] = 0
            edge_counts[service][edge_server] = 0

# Executing the simulation
simulator.run_model()


# Retrieving logs dataframe for plot
logs = pd.DataFrame(simulator.agent_metrics["EdgeServer"])
print(logs)

df = logs

edge_server_ids = df['Instance ID'].unique()

# Determine the number of subplots based on the number of EdgeServers
num_subplots = len(edge_server_ids)  # Add 1 for the rewards subplot
# Create subplots with the desired layout
fig, axes = plt.subplots(num_subplots // 2, 2, figsize=(
    20, 4*num_subplots), sharex=True)
axes = axes.flatten()
# Iterate over each EdgeServer and plot the data in the corresponding subplot
for i, edge_server_id in enumerate(edge_server_ids):
    # Filter the data for the current EdgeServer
    edge_server_data = df[df['Instance ID'] == edge_server_id]

    # Extract the timestep and power consumption values
    timesteps = edge_server_data['Time Step']
    power_consumption = edge_server_data['Power Consumption']

    # Plot the power consumption data for the current EdgeServer in the corresponding subplot
    axes[i].plot(timesteps, power_consumption,
                 label=f"EdgeServer {edge_server_id}")

    # Set the subplot title and labels
    axes[i].set_title(f"Power Consumption - EdgeServer {edge_server_id}")
    axes[i].set_ylabel("Power Consumption")
    axes[i].legend()

plt.tight_layout()

plt.savefig('MAB_migration_power_1_6png')
for j in range(num_subplots, len(axes)):
    fig.delaxes(axes[j])

# Tạo biểu đồ tổng riêng biệt
fig_total, ax_total = plt.subplots(
    figsize=(20, 4))  # Một hàng, toàn chiều rộng
power_count_list = list(range(1, len(power_list) + 1))
ax_total.plot(power_count_list, power_list, label="Total Power")
ax_total.set_title("Total Power")
ax_total.set_xlabel("Timestep Count")
ax_total.set_ylabel("Power")
ax_total.legend()
plt.savefig('MAB_migration_power_total.png')
