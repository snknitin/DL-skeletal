from typing import Union

import torch
import os
import hydra
import omegaconf
import rootutils
import lightning as pl
from torch.utils.data import DataLoader, IterableDataset, ConcatDataset, Dataset, random_split

from src.data.components.PER_buffer import PrioritizedReplayBuffer
from src.data.components.replay_buffer import ReplayBuffer

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: Union[ReplayBuffer, PrioritizedReplayBuffer], sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size
        self.is_per = isinstance(buffer, PrioritizedReplayBuffer)

    def __iter__(self): # -> Tuple:
        if self.is_per:
            batch, indices, weights = self.buffer.sample(self.sample_size)
            states, actions, rewards, dones, next_states = batch
            for i in range(len(dones)):
                yield states[i], actions[i], rewards[i], dones[i], next_states[i], indices[i], weights[i]
        else:
            states, actions, rewards, dones, next_states = self.buffer.sample(self.sample_size)
            for i in range(len(dones)):
                yield states[i], actions[i], rewards[i], dones[i], next_states[i]

class RLDataModule(pl.LightningDataModule):
    def __init__(self, buffer: Union[ReplayBuffer, PrioritizedReplayBuffer], batch_size: int):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = RLDataset(self.buffer)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

if __name__=="__main__":
    pl.seed_everything(3407)
    root = rootutils.setup_root(__file__, pythonpath=True)

    # # Initialize replay buffer
    # buffer = ReplayBuffer(config.data.replay_size)
    #
    # # Initialize data module
    # data_module = RLDataModule(buffer, config.batch_size)

    buffer_cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "buffer.yaml")
    buffer = hydra.utils.instantiate(buffer_cfg)

    # data_cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "rl_data.yaml")
    # data = hydra.utils.instantiate(data_cfg)

    # # Initialize data module
    data_module = RLDataModule(buffer, 200)

    print(data_module)
