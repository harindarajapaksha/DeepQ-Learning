import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optimizer
from collections import deque
import matplotlib.pyplot as plt


""""
Substitute the Bellman equation by the following 
q[state, action] = reward if new state is terminal
                    reward + discount_factor * max(q[new_state,:]) or take the target_dqn(new_input_state).max()
"""



class DQN(nn.Module):
    def __init__(self, input_states, output_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_states, out_features=input_states*4)
        self.fc2 = nn.Linear(in_features=input_states*4, out_features=output_actions*2)
        self.fc3 = nn.Linear(in_features=output_actions*2, out_features=output_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MemoryBuffer():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, x):
        self.memory.append(x)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def return_array(self):
        return self.memory

    def __len__(self):
        return len(self.memory)



class GymEnvironment():
    def __init__(self,render=True):
        mymaps = {
            "Maze": [
                "HHHHHGHHHHH",
                "HFFFFFFFFFH",
                "HGHHHHHHHGH",
                "HFFFFFFFHFH",
                "HHHGHHHGHHH",
                "FFFFFFFFFFF",
                "HHHHHGHHHHH",
                "HFFFFFFFFHH",
                "HHHGHHHGHHH",
                "SFFFFFFFFFF",
                "HHHHHHHHHHH"
            ],
            "Tunnel": [
                "HHHHHHHHHGH",
                "HHHHHHHHHFH",
                "HSFFFFFFFFH",
                "HHHHHHHHHHH",
            ],
            "Tunnel2": [
                "HHHHHHHHHHH",
                "HSFFFFFFFFG",
                "HHHHHHHHHHH"
            ],
            "Tunnel3": [
                "HHHHHHHHHHHHHHHHHHHHH",
                "HSFFFFFFFFFFFFFFFFFGH",
                "HHHHHHHHHHHHHHHHHHHHH"
            ]
        }
        self.env = gym.make(id='FrozenLake-v1',
                            is_slippery=False,
                            render_mode='human' if render == True else "rgb_array"
                            ,desc=mymaps["Tunnel3"]
                            # ,map_name="4x4"
                            )

    def env(self):
        return self.env
    def num_states(self):
        return self.env.observation_space.n

    def num_actions(self):
        return self.env.action_space.n


def format_input_state(num_states, current_state):
    x = torch.zeros(num_states)
    x[current_state] = 1
    return x

def train_policy_network(memory_buffer, sample_size, policydqn, targetdqn,
                         lossfunction, optimizer, discount_factor):

    training_data = memory_buffer.sample(sample_size=sample_size)
    target_q_list = []
    current_q_list = []
    for state_vector, new_state_vector, action, reward, terminated in training_data:
        # Get target value
        if terminated:
            target = torch.FloatTensor([reward])
        else:
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward + discount_factor * targetdqn(new_state_vector).max()
                )

        # Current q valaue
        current_q = policydqn(state_vector)
        current_q_list.append(current_q)

        # Target q values
        target_q = targetdqn(new_state_vector)

        # Update target q value
        target_q[action] = target
        target_q_list.append(target_q)

    # Train the policy
    optimizer.zero_grad()
    loss = lossfunction(torch.stack(current_q_list), torch.stack(target_q_list))
    loss.backward()
    optimizer.step()

    return loss.item()




if __name__ == '__main__':

    render = False
    num_epochs = 10000
    discount_factor = 0.9
    epsilon = 1
    epsilon_minimum = 0.0001
    epsilon_decay = 0.9999
    learning_rate = 0.001
    memory_size = 1024
    batch_size = 64
    sync_rate = 10
    sync_counter = 0



    memory = MemoryBuffer(maxlen=memory_size)
    gymenv = GymEnvironment(render=render)
    num_states = gymenv.num_states()

    policy_dqn = DQN(input_states=num_states, output_actions=gymenv.num_actions())
    target_dqn = DQN(input_states=num_states, output_actions=gymenv.num_actions())

    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer = optimizer.Adam(params=policy_dqn.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    accumulated_rewards = np.zeros(num_epochs)


    while True:

        for epoch in range(num_epochs):
            state = gymenv.env.reset()[0]
            terminated = False
            while not terminated:
                state_vector = format_input_state(num_states=num_states, current_state=state)

                # Deciding to use random or neural network
                if random.random() < epsilon:
                    action = gymenv.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state_vector).argmax().item()

                # Appending the memory buffer
                new_state, reward, terminated, truncated, info = gymenv.env.step(action)
                new_state_vector = format_input_state(num_states=num_states, current_state=new_state)
                memory.append((state_vector, new_state_vector, action, reward, terminated))

                # Start policy network training
                if memory.__len__() > batch_size:
                    batch_loss = train_policy_network(memory_buffer=memory, sample_size=batch_size,
                                         policydqn=policy_dqn, targetdqn=target_dqn, lossfunction=loss_function,
                                         optimizer=optimizer, discount_factor=discount_factor)
                    print(f"Training batch_loss={batch_loss}")
                state = new_state

            accumulated_rewards[epoch] = reward
            # Epsilon decay
            if epsilon > epsilon_minimum and epoch > batch_size*2:
                epsilon = epsilon * epsilon_decay
                if sync_counter > sync_rate:
                    print("Sync")
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    sync_counter = 0


            print(f"Epoch={epoch} epsilon={epsilon} Max rewards={np.sum(accumulated_rewards)} ----------------------------")
            sync_counter += 1
        gymenv.env.close()

        plt.plot(accumulated_rewards)
        plt.show()

        torch.save(policy_dqn, f="Policy.pt")

        if np.sum(accumulated_rewards) > 10:
            break












