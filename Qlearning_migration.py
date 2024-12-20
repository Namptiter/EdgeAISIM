from edge_sim_py import *
import math
import os
import random
import msgpack
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DQN import DQNAgent


# Custom collect method to measure the power consumption of each server
def custom_collect_method(self) -> dict:
    metrics = {
        "Instance ID": self.id,
        "Power Consumption": self.get_power_consumption(),
    }
    return metrics


global agent
global reward_list
global reward_count_list
global reward_count

reward_list = list()
# List to store total power consumption everytime the task scheduling algorithm is used
power_list = list()


def my_algorithm(parameters):

    print("\n\n")
    total_reward = 0
    total_power = 0  # We sum the power consumption after migrating each service

    for service in Service.all():  # Iterate over every service

        # If service needs to be migrated
        if not service.being_provisioned:

            # Initialise our state vector, which is the concatenation of the cpu,memory,disk utilisation and current power consumption

            state_vector = []

            for edge_server in EdgeServer.all():
                edge_server_cpu = edge_server.cpu
                edge_server_memory = edge_server.memory
                edge_server_disk = edge_server.disk
                power = (edge_server_cpu * edge_server_memory *
                         edge_server_disk) ** (1 / 3)
                vector = [edge_server_cpu, edge_server_memory,
                          edge_server_disk, power]
                state_vector = state_vector + vector

            # Pass the state vector to out Q - learning agent, and retrieve action
            state_vector = np.array(state_vector)
            action = agent.choose_action(state_vector)

            # To conserve resources, we don't want to migrate back to our host
            if EdgeServer.all()[action] == service.server:
                break

            print(
                f"[STEP {parameters['current_step']}] Migrating {service} From {service.server} to {EdgeServer.all()[action]}")

            # Migrate service to new edgeserver
            service.provision(target_server=EdgeServer.all()[action])

            # Get our next state, after taking action
            next_state_vector = []
            reward = 0
            power = 0

            for edge_server in EdgeServer.all():
                edge_server_cpu = edge_server.cpu
                edge_server_memory = edge_server.memory
                edge_server_disk = edge_server.disk
                power = (edge_server_cpu * edge_server_memory *
                         edge_server_disk) ** (1 / 3)
                vector = [edge_server_cpu, edge_server_memory,
                          edge_server_disk, power]
                next_state_vector = next_state_vector + vector
                # Our reward is the inverse of the edge server's power consumption
                reward = reward + 1/edge_server.get_power_consumption()
                # get the sum of powerconsumption of each edge server
                power = power + edge_server.get_power_consumption()

            # Pass the updated state and reward for backporpogation through Q - network
            next_state_vector = np.array(next_state_vector)
            agent.update(state_vector, action,
                         next_state_vector, reward, False)

            # print(reward)
            total_reward += reward
            total_power += power  # Sum our power consumption

    reward_list.append(total_reward)
    # Append power consumption to power list for plotting
    power_list.append(total_power)
    # Reduce the probability of agent taking random action for exploration
    agent.epsilon *= agent.epsilon_decay


def stopping_criterion(model: object):
    # As EdgeSimPy will halt the simulation whenever this function returns True,
    # its output will be a boolean expression that checks if the current time step is 600
    return model.schedule.steps == 1000


simulator = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
)

# Loading a sample dataset from GitHub
simulator.initialize(input_file="sample_dataset3.json")

# Assigning the custom collect method
EdgeServer.collect = custom_collect_method

# Initialise of DQN agent with state and action dimension
# Here, state is the current cpu, memory and disk utilisation of the server, and action space is the choice of edge server
# i.e. the Edge server with the maximum Q- value will be migrated to
agent = DQNAgent(len(EdgeServer.all()) * 4, len(EdgeServer.all()))

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

plt.savefig('Qlearning_migration_power_1_6png')
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
plt.savefig("Qlearning_migration_power_total.png")
