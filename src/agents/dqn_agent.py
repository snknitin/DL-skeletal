import numpy as np
import torch
from torch import nn, optim
import hydra
import omegaconf
import rootutils
import lightning as pl
import gym
from src.data.components.replay_buffer import Experience, ReplayBuffer

class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = buffer
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


if __name__=="__main__":
    pl.seed_everything(3407)
    root = rootutils.setup_root(__file__, pythonpath=True)

    # # Initialize replay buffer
    # buffer = ReplayBuffer(config.data.replay_size)
    #
    # # Initialize data module
    # data_module = RLDataModule(buffer, config.batch_size)

    buffer_cfg = omegaconf.OmegaConf.load(root / "configs" / "buffer" / "buffer.yaml")
    buffer = hydra.utils.instantiate(buffer_cfg)

    agent_cfg = omegaconf.OmegaConf.load(root / "configs" / "agent" / "agent.yaml")
    agent = hydra.utils.instantiate(agent_cfg)

    data_cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "rl_data.yaml")
    data = hydra.utils.instantiate(data_cfg)

    # steps = 1000
    # for i in range(steps):
    #     agent.play_step(net, epsilon=1.0)
    #
    # print(agent)