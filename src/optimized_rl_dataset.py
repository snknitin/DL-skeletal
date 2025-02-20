import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Union, Tuple, List, Dict, Optional
import numpy as np


class RLMapDataset(Dataset):
    """
    Map-style Dataset containing samples from the ReplayBuffer

    Args:
        buffer: replay buffer (either regular or prioritized)
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: Union['ReplayBuffer', 'PrioritizedReplayBuffer'], sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size
        self.is_per = hasattr(buffer, 'sample_with_indices')

        # Identify buffer type - check if this is a PrioritizedReplayBuffer by examining return shape
        # Test with a tiny sample first to determine buffer type
        test_sample = buffer.sample(1)
        self.is_per = isinstance(test_sample, tuple) and len(test_sample) == 3
        # Sample the data once upfront
        self._sample_data()

    def _sample_data(self):
        """Sample data from buffer once and store it"""
        if self.is_per:
            batch, self.indices, self.weights = self.buffer.sample(self.sample_size)
            self.states, self.actions, self.rewards, self.dones, self.next_states = batch
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = self.buffer.sample(self.sample_size)
            self.indices = None
            self.weights = None

    def __len__(self) -> int:
        """Return the number of samples"""
        return self.sample_size

    def __getitem__(self, idx: int) -> Union[Tuple, Dict]:
        """Return a single transition"""
        if self.is_per:
            return {
                'state': self.states[idx],
                'action': self.actions[idx],
                'reward': self.rewards[idx],
                'done': self.dones[idx],
                'next_state': self.next_states[idx],
                'indices': self.indices[idx],
                'weights': self.weights[idx]
            }
        else:
            return {
                'state': self.states[idx],
                'action': self.actions[idx],
                'reward': self.rewards[idx],
                'done': self.dones[idx],
                'next_state': self.next_states[idx]
            }


class OptimizedRLDataModule:
    """
    A data module that handles creating datasets and dataloaders for RL training
    with performance optimizations
    """

    def __init__(self,
                 buffer: Union['ReplayBuffer', 'PrioritizedReplayBuffer'],
                 dataset_sample_size: int = 200,
                 batch_size: int = 32,
                 num_workers: int = 0):
        self.buffer = buffer
        self.dataset_sample_size = dataset_sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_dataloader(self) -> DataLoader:
        """Create an optimized dataloader"""
        dataset = RLMapDataset(self.buffer, self.dataset_sample_size)

        # Log info about dataset size for debugging
        print(f"Created dataset with {len(dataset)} samples")

        # Memory usage estimation (rough approximation)
        sample_item = dataset[0]
        sample_size_bytes = sum(tensor.element_size() * tensor.nelement()
                                for tensor in sample_item.values() if torch.is_tensor(tensor))
        estimated_memory_mb = (sample_size_bytes * len(dataset)) / (1024 * 1024)
        print(f"Estimated dataset memory usage: {estimated_memory_mb:.2f} MB")

        # Create dataloader with appropriate settings
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Already sampled randomly from buffer
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

        return dataloader