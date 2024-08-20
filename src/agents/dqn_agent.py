import numpy as np
import torch
from torch import nn, optim
import hydra
import omegaconf
import rootutils
import lightning as pl
import gym
from torch.utils.data import DataLoader

from src.data.components.replay_buffer import Experience, ReplayBuffer
from src.gymenv import MultiFCEnvironment
from src.models.components.dqn_nn import BranchDuelingDQN, BranchDuelingDQNMulti
from src.gymenv.single_env import SingleFCEnvironment
from src.data.rl_datamodule import RLDataset

class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, buffer: ReplayBuffer, seed: int) -> None:
        self.env = env
        self.seed = seed
        self.replay_buffer = buffer
        self.reset()
        self.state = self.env.reset(seed=self.seed)

    def process_state(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            return state[0]
        else:
            return state

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.process_state(self.env.reset(seed=self.seed))

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

        new_state, reward, done, info = self.env.step(action)

        # Process states
        state = np.array(self.process_state(self.state))
        new_state = np.array(self.process_state(new_state))

        exp = Experience(state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done, info


if __name__=="__main__":
    pl.seed_everything(3407)
    root = rootutils.setup_root(__file__, pythonpath=True)
    data_dir = root/"data/item_id_873764"

    # agent_cfg = omegaconf.OmegaConf.load(root / "configs" / "agent" / "agent.yaml")
    # agent_cfg.env.env_cfg.data_dir = data_dir
    # agent = hydra.utils.instantiate(agent_cfg)

    buffer = ReplayBuffer(1000)
    seed = 3407

    env_id= 'SingleFC-v0'
    env_cfg = {
    "item_nbr": 873764,
    "data_dir": data_dir,
    "holding_cost": 0.1,
    "shortage_cost": 1.0,
    "fc_lt_mean": [3],
    "fc_lt_var": [1],
    "num_fc": 16,
    "num_sku": 1,
    "forecast_horizon": 100,
    "starting_week": 12351}

    env = MultiFCEnvironment(env_cfg)
    net = BranchDuelingDQNMulti(obs_size=8*16,n_actions=20,num_fcs=16)
    target_net = BranchDuelingDQNMulti(obs_size=8*16,n_actions=20,num_fcs=16)

    agent = Agent(env,buffer,seed)


    steps = 1000
    for i in range(steps):
        agent.play_step(net, epsilon=1.0)

    #print(agent)
    dataset = RLDataset(buffer, 500)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=150)

    batch = next(iter(dataloader))
    #print(batch)
    states, actions, rewards, dones, next_states = batch
    # Convert actions to torch.int64
    # Ensure actions are long and have the right shape
    actions = actions.long().unsqueeze(-1)  # Shape: (150, 1)

    # Get current Q values
    current_q_values = net(states)  # Shape should be (150, n)

    # Select the Q values for the actions taken
    # Since we only have 1 action outputed we only have 1 q value - change action_dim to 20
    current_q_values = current_q_values.gather(1, actions)  # Shape: (150, 1)
    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        next_q_values[dones] = 0.0
        next_q_values = next_q_values.detach()
        target_q_values = rewards + 0.99 * next_q_values

