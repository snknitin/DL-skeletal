from collections import namedtuple
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn, optim
import hydra
import omegaconf
import rootutils
import lightning as pl
import gym
from torch.utils.data import DataLoader

from src.data.components.PER_buffer import PrioritizedReplayBuffer
from src.data.components.replay_buffer import Experience, ReplayBuffer
from src.gymenv import MultiFCEnvironment
from src.models.components.dqn_nn import BranchDuelingDQN, BranchDuelingDQNMulti
from torchmetrics import MaxMetric, MeanMetric, SumMetric
from src.data.rl_datamodule import RLDataset

import torch
from torch import nn


class MultiItemAgent:
    """
    Agent class handling multiple items with shared buffer
    Args:
        item_ids: List of item IDs to handle
        env_config: Base environment configuration
        buffer: Shared replay buffer
        seed: Random seed
        agent_config: Agent configuration
    """

    def __init__(self, item_ids: List[int],buffer: Union[ReplayBuffer, PrioritizedReplayBuffer], env_config: dict, seed: int, agent_config: dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        torch.manual_seed(seed)

        self.replay_buffer = buffer
        self.item_ids = item_ids

        # Create an agent and environment for each item
        self.agents = {}
        for item_id in item_ids:
            # Modify env config for this item
            item_env_config = env_config.copy()
            item_env_config['item_nbr'] = item_id
            item_env_config['data_dir'] = str(item_env_config['data_dir'])+str(item_id)

            # Create environment and store agent config
            env = MultiFCEnvironment(item_env_config)
            # Set seeds for action and observation spaces
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            base_state = env.reset(seed=seed)
            # Add item features to initial state
            state = self.add_item_features(base_state, item_id)
            self.agents[item_id] = {
                'env': env,
                'state': state,
                'action_low': torch.tensor(agent_config['action_low']),
                'action_high': torch.tensor(agent_config['action_high']),
                'action_segment': torch.tensor(agent_config['action_segment']),
                'act_dim': agent_config['act_dim'],
                'sub_act_dim': agent_config['sub_act_dim']
            }

    def reset(self) -> None:
        """Resets all environments and updates states"""
        for item_id in self.item_ids:
            base_state = self.agents[item_id]['env'].reset(seed=self.seed)
            # Add item features to initial state after reset
            augmented_state = self.add_item_features(base_state, item_id)
            self.agents[item_id]['state'] = augmented_state

    def get_action(self, item_id: int, net: nn.Module, epsilon: float, device: str) -> torch.Tensor:
        """Get action for specific item using epsilon-greedy policy"""
        agent = self.agents[item_id]

        if torch.rand(1).item() < epsilon:
            env_sampled = agent['env'].action_space.sample()
            num_fcs = env_sampled.shape[0]
            action = torch.randint(0, agent['sub_act_dim'], (num_fcs,))
        else:
            with torch.no_grad():
                state = agent['state'].unsqueeze(0).to(device)
                q_values = net(state).squeeze(0)
                action = q_values.argmax(dim=-1).cpu()

        return action

    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu'):
        """
        Carries out a single step for all items and adds experiences to shared buffer
        """
        total_reward = 0
        all_done = True
        info_dict = {}

        for item_id in self.item_ids:
            agent = self.agents[item_id]

            # Get action for this item
            action = self.get_action(item_id, net, epsilon, device)
            action1 = agent['action_low'] + action * agent['action_segment']

            # Get base next_state from environment
            base_next_state, reward, done, info = agent['env'].step(action1)

            # Add item features to next_state
            next_state = self.add_item_features(base_next_state, item_id)

            # Store experience in shared buffer
            exp = Experience(agent['state'], action, reward, done, next_state)
            self.replay_buffer.append(exp)

            agent['state'] = next_state
            if done:
                base_state = agent['env'].reset(seed=self.seed)
                agent['state'] = self.add_item_features(base_state, item_id)


            # Update aggregated values
            total_reward += reward
            all_done = all_done and done
            info_dict[item_id] = info

        return total_reward, all_done, info_dict

    def add_item_features(self, state: torch.Tensor, item_id: int) -> torch.Tensor:
        """
        Add item-specific features to state before passing to network
        """
        # Option 1: Concatenate one-hot encoded item_id
        item_idx = self.item_ids.index(item_id)
        one_hot = torch.zeros(len(self.item_ids), device=self.device)
        one_hot[item_idx] = 1

        # If state is not already a tensor, convert it
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)

        # Ensure state is on the correct device
        state = state.to(self.device)

        # Concatenate state and one-hot encoding
        return torch.cat([state, one_hot])

        # Option 2: Add learned item embeddings from descriptions, if available
        # item_embedding = self.item_embeddings[item_id]
        # return torch.cat([state, item_embedding.expand(state.size(0), -1)], dim=1)

        # Option 3: Add static item features
        # item_features = self.item_feature_dict[item_id]  # price, lead time, etc.
        # return torch.cat([state, item_features.expand(state.size(0), -1)], dim=1)

        # return torch.cat([state, one_hot.expand(state.size(0), -1)], dim=1)

if __name__=="__main__":
    pl.seed_everything(340)
    root = rootutils.setup_root(__file__, pythonpath=True)
    data_dir = root/"data/item_id_"

    # agent_cfg = omegaconf.OmegaConf.load(root / "configs" / "agent" / "agent.yaml")
    # agent_cfg.env.env_cfg.data_dir = data_dir
    # agent = hydra.utils.instantiate(agent_cfg)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = ReplayBuffer(10000)
    seed = 3407

    env_id= 'MultiFC_OT-v0'
    item_ids = [873764,782523046,873685,36717960,391706535,167121557]  # List of item IDs
    item_feature_size = len(item_ids)

    env_cfg = {
    "data_dir": data_dir,
    "holding_cost": 0.1,
    "shortage_cost": 1.0,
    "sel_FCs": ["4270", "6284", "6753"],   #"4364","4401"],
    "num_fcs": 3,
    "num_sku": 1,
    "forecast_horizon": 100,
    "starting_week": 12351}

    agent_config = {
    "action_low": 0,
    "action_high": 3,
    "action_segment": 0.15,
    "act_dim": 3,
    "sub_act_dim": 20}

    num_fcs = env_cfg.get("num_fcs",16)


    agent = MultiItemAgent(
            item_ids=item_ids,
            env_config=env_cfg,
            buffer=buffer,
            seed=seed,
            agent_config=agent_config
        )

    input_size = 25

    net = BranchDuelingDQNMulti(obs_size=input_size,n_actions=20,num_fcs=num_fcs,item_feature_size=item_feature_size)
    target_net = BranchDuelingDQNMulti(obs_size=input_size,n_actions=20,num_fcs=num_fcs,item_feature_size=item_feature_size)



    steps = 1000
    for i in range(steps):
        # print("step",i)
        agent.play_step(net, epsilon=0.9)
    print("Done filling the buffer for warm start")


    #print(agent)
    dataset = RLDataset(buffer, 500)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=150)

    train_loss = MeanMetric()

    for _ in range(10000):
        batch = next(iter(dataloader))
        #print(batch)
        states, actions, rewards, dones, next_states = batch
        # Convert actions to torch.int64
        # Ensure actions are long and have the right shape
        actions = actions.long() # Shape: (150, num_fcs)

        # Get current Q values
        current_q_values = net(states)  # Shape should be (150, fc, n)

        # Select the Q values for the actions taken
        # Since we only have 1 action outputed we only have 1 q value - change action_dim to 20
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, num_fcs)
        # Compute target Q values
        with torch.no_grad():
            next_q_values = target_net(next_states).max(dim=-1)[0]
            next_q_values[dones] = 0.0

            # # For Vector Reward: ===========================================================
            next_q_values = next_q_values.detach() # Shape: (batch_size, num_fcs)
            target_q_values = rewards.unsqueeze(1) + 0.99 * next_q_values # Shape: (batch_size, num_fcs)
            ##-------------------------------------------------------------------------------
            # For Scalar Reward
            # next_q_values = next_q_values.detach().mean(1) # Shape: (batch_size)
            # target_q_values = rewards + 0.99 * next_q_values # Shape: (batch_size)
            # target_q_values = target_q_values.unsqueeze(1).repeat(1, num_fcs) # Shape: (batch_size, num_fcs)
            ## ================================================================================

        loss = nn.MSELoss()(current_q_values, target_q_values)
        train_loss(loss.item())
        print(loss)

    print(train_loss.compute())