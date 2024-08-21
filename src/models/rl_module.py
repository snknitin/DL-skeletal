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
from src.data.rl_datamodule import RLDataModule, RLDataset
from collections import OrderedDict
from src.models.components.dqn_nn import DQN
from torchmetrics import MaxMetric, MeanMetric, SumMetric

from src import gymenv  # this is enough to register the environment


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self,env: str, seed: int, net: DQN, target_net: DQN, buffer,optimizer,
                 eps_start: float, eps_end: float, eps_last_frame: int,
                 sync_rate: int, lr: float, gamma: float, warm_start_steps:int,
                 dataset_sample_size: int,batch_size: int) -> None:
        super().__init__()

        self.save_hyperparameters()

        # self.env = gym.make(self.hparams.env)
        self.env = gym.make(self.hparams.env['id'], env_cfg=self.hparams.env['env_cfg'])

        # Need to set these seeds too
        self.env.action_space.seed(self.hparams.seed)
        self.env.observation_space.seed(self.hparams.seed)

        # obs_size = self.env.observation_space.shape[0]
        # n_actions = self.env.action_space.n

        self.net = self.hparams.net
        self.target_net = self.hparams.target_net

        self.buffer = self.hparams.buffer
        self.agent = Agent(self.env, self.buffer,self.hparams.seed)

        self.total_reward = 0
        self.episode_reward = 0


        self.lr = lr
        print(self.lr)

        # Custom counters
        self.episode_count = 0
        self.step_count = 0

        self.populate(self.hparams.warm_start_steps)

        # Metrics
        # for averaging loss across batches - epochs/episodes

        self.train_loss = MeanMetric()
        self.avg_episodic_reward = MeanMetric()

        self.episode_reward = SumMetric()
        self.episode_rewards = [0]
        self.cumulative_step_reward = 0
        self.cumulative_episode_reward = 0
        self.mavg_reward = 0

        self.episode_length = SumMetric()

        self.avg_holding_cost = MeanMetric()
        self.avg_shortage_cost = MeanMetric()

        self.total_holding_cost = SumMetric()
        self.total_shortage_cost = SumMetric()

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

    def dqn_mse_loss(self, batch) -> torch.Tensor: # : Tuple[torch.Tensor, torch.Tensor]
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        # Convert actions to torch.int64
        # Ensure actions are long and have the right shape
        actions = actions.long()  # Shape: (150, num_fc)

        # Get current Q values
        current_q_values = self.net(states)  # Shape should be (150, num_fc, n_action)

        # Select the Q values for the actions taken
        # Since we only have 1 action outputed we only have 1 q value - change action_dim to 20
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # Shape: (150, 1)
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=-1)[0]
            next_q_values[dones] = 0.0
            next_q_values = next_q_values.detach()
            target_q_values = rewards + self.hparams.gamma * next_q_values

        return nn.MSELoss()(current_q_values.squeeze(), target_q_values)

    def on_train_start(self):
        # To ensure every run is proper from start to finish and not mid-way from populate buffer
        self.env.reset()
        # Reset metrics for next episode

        self.train_loss.reset()
        self.avg_episodic_reward.reset()

        self.avg_holding_cost.reset()
        self.total_holding_cost.reset()
        self.avg_shortage_cost.reset()
        self.total_shortage_cost.reset()

        # Reset metrics for next episode
        self.episode_reward.reset()
        self.episode_length.reset()

        self.episode_rewards = [0]
        self.cumulative_step_reward = 0

    def training_step(self, batch, nb_batch): # : Tuple[torch.Tensor, torch.Tensor]
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        self.step_count += 1
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      ((self.step_count + 1) / (self.hparams.eps_last_frame)))

        # step through environment with agent
        reward, done, info = self.agent.play_step(self.net, epsilon, self.device)
        # self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        # update loss and log
        self.train_loss(loss)
        # Update both reward metrics
        self.cumulative_step_reward += reward.item()
        self.episode_reward(reward)
        self.avg_episodic_reward(reward)

        # Log moving averages
        window_size = min(5, len(self.episode_rewards))
        mavg_reward = sum(self.episode_rewards[-window_size:]) / window_size
        cumulative_episode_reward = sum(self.episode_rewards)



        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/epsilon", epsilon, on_step=False, on_epoch=True, prog_bar=True)

        self.log("train/cumulative_step_reward", self.cumulative_step_reward, on_step=False, on_epoch=True)
        self.log("train/cumulative_episodic_reward", cumulative_episode_reward, on_step=False, on_epoch=True)

        self.log("train/Moving_avg_5_ep_reward", mavg_reward, on_step=False, on_epoch=True)


        # Update metrics
        self.episode_length(1)  # Increment by 1 for each step
        # Update metrics
        self.avg_holding_cost(info['holding_cost'])
        self.total_holding_cost(info['holding_cost'])
        self.avg_shortage_cost(info['shortage_cost'])
        self.total_shortage_cost(info['shortage_cost'])

        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.step_count % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # total_reward = torch.tensor(self.total_reward,dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward,dtype=torch.float32).to(self.device).item()
        steps = torch.tensor(self.global_step).to(self.device)

        log = {'loss': loss,
               'reward': reward,
               'steps': steps}


        # Log step-level metrics
        # Log environment step metrics
        self.log("env_step/reward", reward, on_step=False, on_epoch=True, prog_bar=False)
        self.log("env_step/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/inv_at_day_start', info["inv_at_beginning_of_day"], on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('env_step/inv_after_replen', info["inv_after_replen"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/inv_at_end_of_day', info["inv_at_end_of_day"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/action', info["action"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/repl_rec', info["repl_rec"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/sales_at_FC', info["sales_at_FC"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/holding_cost', info["holding_cost"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/shortage_cost', info["shortage_cost"], on_step=False, on_epoch=True, prog_bar=False)
        self.log('env_step/mapped_dem', info["mapped_dem"], on_step=False, on_epoch=True, prog_bar=False)

        # End of episode
        if done:
            self.episode_count += 1
            episode_reward = self.episode_reward.compute()
            episode_avg_reward = self.avg_episodic_reward.compute()
            self.episode_rewards.append(episode_reward.item())


            # Log episode-level metrics only when an episode is done
            self.log("episode/total_reward", episode_reward.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("episode/avg_reward", episode_avg_reward.item(), on_step=False, on_epoch=True, prog_bar=False)

            self.log("episode/length", self.episode_length.compute(), on_step=False, on_epoch=True)
            self.log("episode/avg_loss", self.train_loss.compute(), on_step=False, on_epoch=True)
            self.log("episode/avg_holding_cost", self.avg_holding_cost.compute(), on_step=False, on_epoch=True)
            self.log("episode/total_holding_cost", self.total_holding_cost.compute(), on_step=False, on_epoch=True)
            self.log("episode/avg_shortage_cost", self.avg_shortage_cost.compute(), on_step=False, on_epoch=True)
            self.log("episode/total_shortage_cost", self.total_shortage_cost.compute(), on_step=False, on_epoch=True)

            # Log custom episode count
            self.log("episode/count", self.episode_count, on_step=False, on_epoch=True)

            # Reset metrics for next episode
            self.avg_holding_cost.reset()
            self.total_holding_cost.reset()
            self.avg_shortage_cost.reset()
            self.total_shortage_cost.reset()

            # Reset metrics for next episode
            self.episode_reward.reset()
            self.avg_episodic_reward.reset()
            self.episode_length.reset()

            # Reset the environment for the next episode
            self.env.reset()

        return {'loss': loss, 'reward': reward}


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(),lr=self.lr)
        # cycle momentum needs to be False for Adam to work
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer = optimizer,
        #                                          base_lr = self.lr,
        #                                          max_lr= 0.5,
        #                                           step_size_up = 35,
        #                                           step_size_down = 40,
        #                                           cycle_momentum= False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.trainer.max_epochs,0)
        return [optimizer] , [lr_scheduler]

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.dataset_sample_size)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                # num_workers=4,  # Adjust this based on your system
                                # pin_memory=True if torch.cuda.is_available() else False,
                                # persistent_workers = True
                                )
        return dataloader



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
    # trainer.save_checkpoint("example.ckpt")
