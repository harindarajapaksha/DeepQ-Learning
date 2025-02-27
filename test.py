import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F


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
                            ,desc=mymaps["Tunnel2"]
                            # ,map_name="4x4"
                            )
    def env(self):
        return self.env
    def num_states(self):
        return self.env.observation_space.n

    def num_actions(self):
        return self.env.action_space.n



def format_input_state(current_state, num_states):
    x = torch.zeros(num_states)
    x[current_state] = 1
    return x

if __name__ == '__main__':

    gymenv = GymEnvironment()
    num_epoches = 100
    num_states = gymenv.num_states()

    policydqn = torch.load(f="Policy_tunnel2_2.pt", weights_only=False)
    policydqn.eval()

    for epoch in range(num_epoches):
        state = gymenv.env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            with torch.no_grad():
                action = policydqn(format_input_state(current_state=state, num_states=num_states)).argmax().item()
            state, reward, terminated, truncated, _ = gymenv.env.step(action)

    gymenv.env.close()

