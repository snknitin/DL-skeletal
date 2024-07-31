import numpy as np
import torch
from torch import nn, optim
import gym
from src.data.components.replay_buffer import Experience, ReplayBuffer

class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def process_state(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            return state[0]
        else:
            return state

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.process_state(self.env.reset())

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = self.process_state(self.state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = state.to(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu'): # -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        new_state, reward, done, _, _ = self.env.step(action)

        # Process states
        state = np.array(self.process_state(self.state))
        new_state = np.array(self.process_state(new_state))

        exp = Experience(state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done