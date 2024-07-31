import os
import hydra
import omegaconf
import rootutils
import gym
import torch
from torch import nn, optim
import lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from src.agents.dqn_agent import Agent
from src.data.rl_datamodule import RLDataModule, RLDataset
from collections import OrderedDict
from src.models.components.dqn_nn import DQN
from torchmetrics import MaxMetric, MeanMetric

class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self,env: str, net: DQN, target_net: DQN, buffer,optimizer,
                 eps_start: float, eps_end: float, eps_last_frame: int,
                 sync_rate: int, lr: float, gamma: float, warm_start_steps:int,
                 episode_length: int,batch_size: int) -> None:
        super().__init__()

        self.save_hyperparameters()

        # self.hparams = hparams

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = self.hparams.net
        self.target_net = self.hparams.target_net

        self.buffer = self.hparams.buffer
        self.agent = Agent(self.env, self.buffer)

        self.total_reward = 0
        self.episode_reward = 0

        self.lr = lr

        self.populate(self.hparams.warm_start_steps)
        # for averaging loss across batches
        self.train_loss = MeanMetric()

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
        actions = actions.long()
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch, nb_batch) -> OrderedDict: # : Tuple[torch.Tensor, torch.Tensor]
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      self.global_step + 1 / self.hparams.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, self.device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        total_reward = torch.tensor(self.total_reward,dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        steps = torch.tensor(self.global_step).to(self.device)

        log = {'total_reward': total_reward,
               'reward': reward,
               'steps': steps}

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("reward", reward.item(), on_step=True, on_epoch=False, prog_bar=False)
        self.log("total_reward", total_reward.item(), on_step=False, on_epoch=True, prog_bar=True)

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})

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
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
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

    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "RL_inv.yaml")
    model = hydra.utils.instantiate(model_cfg)


    trainer = pl.Trainer(
            # gpus=1,
            # distributed_backend='dp',
            max_epochs=1000,
            #early_stop_callback=False,
            val_check_interval=100
        )

    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")
