import math
import os
import hydra
import omegaconf
import rootutils
import gym
import torch
from lightning.pytorch.tuner import Tuner
from torch import nn, optim
import lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from src.agents.dqn_agent import Agent
from src.agents.multi_item_agent import MultiItemAgent
from src.data.components.PER_buffer import PrioritizedReplayBuffer
from src.data.rl_datamodule import RLDataModule, RLDataset
from collections import OrderedDict
from src.models.components.dqn_nn import DQN
from src.data.components.replay_buffer import Experience, ReplayBuffer

from torchmetrics import MaxMetric, MeanMetric, SumMetric

from src import gymenv  # this is enough to register the environment



class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, env: str, seed: int, num_fcs: int, item_ids, net: DQN, target_net: DQN, optimizer,
                 eps_start: float, eps_end: float, eps_last_frame: int, capacity: int, buffer_type: str,
                 sync_rate: int, lr: float, log_fc_metrics: bool, agent_config: dict, per_params: dict, gamma: float,
                 warm_start_steps: int,dataset_sample_size: int, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters()

        # self.env = gym.make(self.hparams.env)
        self.num_fcs = self.hparams.num_fcs

        #self.env = gym.make(self.hparams.env['id'], env_cfg=self.hparams.env['env_cfg'])

        self.lr = lr
        print(self.lr)
        self.item_ids = item_ids
        print("Running Shared Policy Network for items : ",self.item_ids)
        self.log_fc_metrics = log_fc_metrics



        # self.save_hyperparameters(ignore=['net','target_net'])


        # self.buffer = ReplayBuffer(self.hparams.capacity)
        # Initialize appropriate buffer
        if buffer_type == "regular":
            self.buffer = ReplayBuffer(self.hparams.capacity)
        else:
            self.buffer = PrioritizedReplayBuffer(
                self.hparams.capacity,
                alpha=self.hparams.per_params['alpha'],
                beta=self.hparams.per_params['beta'],
                beta_increment=self.hparams.per_params['beta_increment'],
                epsilon=self.hparams.per_params['epsilon']
            )

        self.agent = MultiItemAgent(
            item_ids=self.item_ids,
            env_config=self.hparams.env['env_cfg'],
            buffer=self.buffer,
            seed=self.hparams.seed,
            agent_config=self.hparams.agent_config
        )

        print(f"Initialized Multi-Item, Multi-FC Agent for {len(self.item_ids)} items and {self.num_fcs} FCs" )
        # Moved this into the multiagent class
        # Need to set these seeds too
        # self.env.action_space.seed(self.hparams.seed)
        # self.env.observation_space.seed(self.hparams.seed)

        self.net = self.hparams.net
        self.target_net = self.hparams.target_net
        # self.agent = Agent(self.env, self.buffer, self.hparams.seed, self.hparams.agent_config)
        self.populate(self.hparams.warm_start_steps)


        # Custom counters
        self.episode_count = 0
        self.total_reward = 0
        self.episode_reward = 0

        #### Metrics ####
        # =================
        # for averaging loss across batches - epochs/episodes
        self.train_loss = MeanMetric()
        self.avg_episodic_reward = MeanMetric()
        # Reward metrics
        self.episode_reward = SumMetric()
        self.episode_rewards = [0]
        self.cumulative_step_reward = 0
        self.cumulative_episode_reward = 0
        self.mavg_reward = 0

        self.episode_length = SumMetric()

        # Cost based metrics
        self.avg_holding_qty_ot = MeanMetric()
        self.avg_shortage_cost_ot = MeanMetric()

        self.total_holding_cost_ot = SumMetric()
        self.total_shortage_cost_ot = SumMetric()

        # Per-FC metrics
        self.fc_rewards_avg = [MeanMetric() for _ in range(self.num_fcs)]
        self.fc_holding_costs_pr_avg = [MeanMetric() for _ in range(self.num_fcs)]
        self.fc_shortage_costs_pr_avg = [MeanMetric() for _ in range(self.num_fcs)]

        self.fc_rewards_sum = [SumMetric() for _ in range(self.num_fcs)]
        self.fc_holding_costs_pr_sum = [SumMetric() for _ in range(self.num_fcs)]
        self.fc_shortage_costs_pr_sum = [SumMetric() for _ in range(self.num_fcs)]

        # checking sync across all the places it is sent
        assert self.buffer is self.agent.replay_buffer, "Buffer Mismatch!"

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch) -> torch.Tensor:  # : Tuple[torch.Tensor, torch.Tensor]
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        if self.hparams.buffer_type == "prioritized":
            states, actions, rewards, dones, next_states, indices, weights = batch
            actions = actions.long()

            current_q_values = self.net(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(dim=-1)[0]
                next_q_values[dones] = 0.0
                next_q_values = next_q_values.detach().mean(1)
                target_q_values = rewards + self.hparams.gamma * next_q_values
                target_q_values = target_q_values.unsqueeze(1).repeat(1, self.num_fcs)

            # Calculate TD errors for priority update
            td_errors = (current_q_values - target_q_values).pow(2).mean(1)

            # Apply weights to squared errors before mean
            loss = (td_errors * weights).mean()

            # Update priorities using TD errors
            priorities = td_errors.detach().cpu().numpy() + 1e-5
            self.buffer.update_priorities(indices, priorities)

            return loss

        else:
            states, actions, rewards, dones, next_states = batch
            # Convert actions to torch.int64
            # Ensure actions are long and have the right shape
            actions = actions.long()  # Shape: (150, num_fc)

            # Get current Q values
            current_q_values = self.net(states)  # Shape should be (150, num_fc, n_action) #n_action -> sub_action_dim

            # Select the Q values for the actions taken
            # Since we only have 1 action outputed we only have 1 q value - change action_dim to 20
            current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # Shape: (150, 1)
            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(dim=-1)[0]
                #Karthik - Commenting the line below to NOT set the next_q_values to 0 at the end of episode
                #next_q_values[dones] = 0.0

                # next_q_values = next_q_values.detach() # Shape: (batch_size, num_fcs)
                # target_q_values = rewards.unsqueeze(1) + self.hparams.gamma * next_q_values # Shape: (batch_size, num_fcs)

                # ## For Scalar Reward: ===========================================
                next_q_values = next_q_values.detach().mean(1)  # Shape: (batch_size)
                target_q_values = rewards + self.hparams.gamma * next_q_values  # Shape: (batch_size)
                target_q_values = target_q_values.unsqueeze(1).repeat(1, self.num_fcs)  # Shape: (batch_size, num_fcs)
                # ## ==============================================================

            return nn.MSELoss()(current_q_values, target_q_values)


    def on_train_start(self):
        # To ensure every run is proper from start to finish and not mid-way from populate buffer
        # self.env.reset()
        self.agent.reset()

        # Reset metrics for next episode

        self.train_loss.reset()
        self.avg_episodic_reward.reset()

        # Reset per-FC metrics
        for fc in range(self.num_fcs):
            self.fc_rewards_avg[fc].reset()
            self.fc_holding_costs_pr_avg[fc].reset()
            self.fc_shortage_costs_pr_avg[fc].reset()
            self.fc_rewards_sum[fc].reset()
            self.fc_holding_costs_pr_sum[fc].reset()
            self.fc_shortage_costs_pr_sum[fc].reset()

        self.avg_holding_qty_ot.reset()
        self.total_holding_cost_ot.reset()
        self.avg_shortage_cost_ot.reset()
        self.total_shortage_cost_ot.reset()

        # Reset metrics for next episode
        self.episode_reward.reset()
        self.episode_length.reset()

        self.episode_rewards = [0]
        self.cumulative_step_reward = 0

    def training_step(self, batch, nb_batch):  # : Tuple[torch.Tensor, torch.Tensor]
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        # epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
        #               ((self.global_step + 1) / (self.hparams.eps_last_frame)))


        # Exponential decay for epsilon
        # epsilon = self.hparams.eps_end + (self.hparams.eps_start - self.hparams.eps_end) * \
        #           math.exp(-1. * (self.global_step + 1) / self.hparams.eps_last_frame)

        epsilon = max(self.hparams.eps_end, (self.hparams.eps_start - self.hparams.eps_end) * math.exp(-1. * (self.global_step + 1) / self.hparams.eps_last_frame))

        # step through environment with agent
        reward_vec, done, info_dict = self.agent.play_step(self.net, epsilon, self.device)
        reward = reward_vec.sum()  # Only for plotting


        loss = self.dqn_mse_loss(batch)


        # update loss and log
        self.train_loss(loss)

        # Update both reward metrics
        self.cumulative_step_reward += reward.item()
        self.episode_reward(reward)
        self.avg_episodic_reward(reward)
        # Update metrics
        self.episode_length(1)  # Increment by 1 for each step

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/epsilon", epsilon, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/cumulative_step_reward", self.cumulative_step_reward, on_step=False, on_epoch=True)

        # Log moving averages
        window_size = min(5, len(self.episode_rewards))
        mavg_reward = sum(self.episode_rewards[-window_size:]) / window_size
        cumulative_episode_reward = sum(self.episode_rewards)
        self.log("train/cumulative_episodic_reward", cumulative_episode_reward, on_step=False, on_epoch=True)
        self.log("train/Moving_avg_5_ep_reward", mavg_reward, on_step=False, on_epoch=True)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # total_reward = torch.tensor(self.total_reward,dtype=torch.float32).to(self.device)
        # reward = torch.tensor(reward, dtype=torch.float32).to(self.device).item()
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)


        reward = reward.clone().detach().to(self.device).item()

        steps = torch.tensor(self.global_step).to(self.device)

        log = {'loss': loss,
               'reward': reward,
               'steps': steps}

        # Single loop for all item and FC specific operations
        for item_id in self.item_ids:
            item_info = info_dict[item_id]

            # Update OT metrics
            self.avg_holding_qty_ot(item_info['holding_qty_ot'].sum())
            self.total_holding_cost_ot(item_info['holding_qty_ot'].sum())
            self.avg_shortage_cost_ot(item_info['shortage_qty_ot'].sum())
            self.total_shortage_cost_ot(item_info['shortage_qty_ot'].sum())

            # Environment step metrics with item_id
            self.log(f"env_step/item_{item_id}/reward", item_info['reward'].sum(), on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"env_step/item_{item_id}/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"env_step/item_{item_id}/total_holding_cost_ot_fcs", item_info['holding_qty_ot'].sum().item(),
                     on_step=False, on_epoch=True)
            self.log(f"env_step/item_{item_id}/total_shortage_cost_ot_fcs", item_info['shortage_qty_ot'].sum().item(),
                     on_step=False, on_epoch=True)

            # FC-specific metrics and logging with item_id
            for fc in range(self.num_fcs):
                # Update metrics
                self.fc_holding_costs_pr_avg[fc](item_info['holding_qty_pr'][fc])
                self.fc_shortage_costs_pr_avg[fc](item_info['shortage_qty_pr'][fc])
                self.fc_holding_costs_pr_sum[fc](item_info['holding_qty_pr'][fc])
                self.fc_shortage_costs_pr_sum[fc](item_info['shortage_qty_pr'][fc])

                if self.log_fc_metrics:
                    self.log(f"fc_{fc}/item_{item_id}/holding_qty_pr", item_info['holding_qty_pr'][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/shortage_qty_pr", item_info['shortage_qty_pr'][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/inv_at_day_start", item_info["inv_at_beginning_of_day"][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/inv_after_replen", item_info["inv_after_replen"][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/inv_at_end_of_day", item_info["inv_at_end_of_day"][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/action", item_info["action"][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/repl_rec", item_info["repl_rec"][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/sales_at_FC", item_info["sales_at_FC"][fc],
                             on_step=False, on_epoch=True)
                    self.log(f"fc_{fc}/item_{item_id}/mapped_dem", item_info["mapped_dem"][fc],
                             on_step=False, on_epoch=True)



        # End of episode
        if done:
            self.episode_count += 1
            episode_reward = self.episode_reward.compute()
            episode_avg_reward = self.avg_episodic_reward.compute()
            self.episode_rewards.append(episode_reward.item())

            # Log episode metrics with item_id
            for item_id in self.item_ids:
                self.log(f"episode/item_{item_id}/total_reward", episode_reward.item(),
                         on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"episode/item_{item_id}/avg_reward", episode_avg_reward.item(),
                         on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"episode/item_{item_id}/length", self.episode_length.compute(),
                         on_step=False, on_epoch=True)
                self.log(f"episode/item_{item_id}/avg_loss", self.train_loss.compute(),
                         on_step=False, on_epoch=True)
                self.log(f"episode/item_{item_id}/avg_holding_qty_ot", self.avg_holding_qty_ot.compute(),
                         on_step=False, on_epoch=True)
                self.log(f"episode/item_{item_id}/total_holding_qty_ot", self.total_holding_cost_ot.compute(),
                         on_step=False, on_epoch=True)
                self.log(f"episode/item_{item_id}/avg_shortage_qty_ot", self.avg_shortage_cost_ot.compute(),
                         on_step=False, on_epoch=True)
                self.log(f"episode/item_{item_id}/total_shortage_qty_ot", self.total_shortage_cost_ot.compute(),
                         on_step=False, on_epoch=True)

                # Episode FC-specific metrics with item_id
                if self.log_fc_metrics:
                    for fc in range(self.num_fcs):
                        self.log(f"episode/item_{item_id}/fc_{fc}/avg_reward",
                                 self.fc_rewards_avg[fc].compute(), on_step=False, on_epoch=True)
                        self.log(f"episode/item_{item_id}/fc_{fc}/avg_holding_qty_pr",
                                 self.fc_holding_costs_pr_avg[fc].compute(), on_step=False, on_epoch=True)
                        self.log(f"episode/item_{item_id}/fc_{fc}/avg_shortage_qty_pr",
                                 self.fc_shortage_costs_pr_avg[fc].compute(), on_step=False, on_epoch=True)
                        self.log(f"episode/item_{item_id}/fc_{fc}/total_reward",
                                 self.fc_rewards_sum[fc].compute(), on_step=False, on_epoch=True)
                        self.log(f"episode/item_{item_id}/fc_{fc}/total_holding_qty_pr",
                                 self.fc_holding_costs_pr_sum[fc].compute(), on_step=False, on_epoch=True)
                        self.log(f"episode/item_{item_id}/fc_{fc}/total_shortage_qty_pr",
                                 self.fc_shortage_costs_pr_sum[fc].compute(), on_step=False, on_epoch=True)

            # Log custom episode count
            # self.log("episode/count", self.episode_count, on_step=False, on_epoch=True)

            # Reset per-FC metrics
            for fc in range(self.num_fcs):
                self.fc_rewards_avg[fc].reset()
                self.fc_holding_costs_pr_avg[fc].reset()
                self.fc_shortage_costs_pr_avg[fc].reset()
                self.fc_rewards_sum[fc].reset()
                self.fc_holding_costs_pr_sum[fc].reset()
                self.fc_shortage_costs_pr_sum[fc].reset()

            # Reset metrics for next episode
            self.avg_holding_qty_ot.reset()
            self.total_holding_cost_ot.reset()
            self.avg_shortage_cost_ot.reset()
            self.total_shortage_cost_ot.reset()

            # Reset metrics for next episode
            self.episode_reward.reset()
            self.avg_episodic_reward.reset()
            self.episode_length.reset()

            # Reset the environment for the next episode
            #self.env.reset()
            self.agent.reset()

        return {'loss': loss, 'reward': reward}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(), lr=self.lr)
        # cycle momentum needs to be False for Adam to work
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer = optimizer,
        #                                          base_lr = self.lr,
        #                                          max_lr= 0.5,
        #                                           step_size_up = 35,
        #                                           step_size_down = 40,
        #                                           cycle_momentum= False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [lr_scheduler]

    # def test_step(self, batch, batch_idx):
    #     self.net.eval()
    #     total_steps = 60  # test data size = 60 days
    #     state = self.env.reset(options={'test': True})
    #     self.agent.state = state  # Update agent's state
    #
    #     for i in range(total_steps):
    #         reward, done, info = self.agent.play_step(self.net, epsilon=0, device=self.device)
    #
    #         # Log metrics
    #         self.log("test/inv_after_replen", sum(info['inv_after_replen']), on_step=False, on_epoch=True)
    #         self.log("test/mapped_dem", sum(info['mapped_dem']), on_step=False, on_epoch=True)
    #         self.log("test/inv_at_end_of_day", sum(info['inv_at_end_of_day']), on_step=False, on_epoch=True)
    #
    #         for fc in range(self.num_fcs):
    #             self.log(f"test/fc_{fc}/inv_after_replen", info['inv_after_replen'][fc], on_step=False, on_epoch=True)
    #             self.log(f"test/fc_{fc}/shortage_cost", info['shortage_cost'][fc], on_step=False, on_epoch=True)
    #             self.log(f"test/fc_{fc}/holding_cost", info['holding_cost'][fc],on_step=False, on_epoch=True)
    #             self.log(f"test/fc_{fc}/mapped_dem", info['mapped_dem'][fc], on_step=False, on_epoch=True)
    #
    #
    #         if done:
    #             break
    #
    #     return None


    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.dataset_sample_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                # num_workers=1, # Adjust this based on your system
                                # pin_memory=True if torch.cuda.is_available() else False,
                                # persistent_workers = True
                                )
        # print("Reloaded")
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def val_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'


if __name__=="__main__":
    pl.seed_everything(3407)
    root = rootutils.setup_root(__file__, pythonpath=True)

    # # Initialize replay buffer
    # buffer = ReplayBuffer(config.data.replay_size)
    #
    # # Initialize data module
    # data_module = RLDataModule(buffer, config.batch_size)

    # buffer_cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "buffer.yaml")
    # buffer = hydra.utils.instantiate(buffer_cfg)

    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "Single_FC.yaml")
    model = hydra.utils.instantiate(model_cfg)


    trainer = pl.Trainer(accelerator='cpu',
            # gpus=1,
            # distributed_backend='dp',
            max_epochs=1000,
            #early_stop_callback=False,
            val_check_interval=100
        )

    # Create a Tuner
    tuner = Tuner(trainer)
    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    tuner.lr_find(model, max_lr=0.9, min_lr=1e-7)
    # Auto-scale batch size with binary search
    # tuner.scale_batch_size(model, mode="binsearch")
    print("Tuned Learning Rate is :", model.hparams.lr)
    print("Tuned Learning Rate is :", model.lr)



    # trainer.fit(model)
    # train