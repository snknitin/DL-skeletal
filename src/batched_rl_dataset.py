import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Dict, Tuple
import numpy as np


class BatchedRLDataset(Dataset):
    """
    Map-style Dataset that pre-organizes buffer samples into batches
    for more efficient access and better memory locality.

    This implementation:
    1. Samples from buffer once at initialization
    2. Pre-organizes samples into batch-sized chunks
    3. Returns entire batches at once, reducing individual access overhead
    """

    def __init__(self, buffer, sample_size=1000, batch_size=32):
        self.buffer = buffer
        self.sample_size = sample_size
        self.batch_size = batch_size

        # Calculate number of batches
        self.num_batches = (sample_size + batch_size - 1) // batch_size

        # Sample and prepare all batches upfront
        self._prepare_batches()

    def _prepare_batches(self):
        """Sample from buffer and prepare batched data structure"""
        # Identify buffer type
        test_sample = self.buffer.sample(1)
        self.is_per = isinstance(test_sample, tuple) and len(test_sample) == 3

        # Sample all required data at once
        if self.is_per:
            batch, indices, weights = self.buffer.sample(self.sample_size)
            states, actions, rewards, dones, next_states = batch
        else:
            states, actions, rewards, dones, next_states = self.buffer.sample(self.sample_size)
            indices, weights = None, None

        # Pre-organize into batches for efficient retrieval
        self.batched_data = []
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.sample_size)
            actual_batch_size = end_idx - start_idx

            # Skip creating empty batches
            if actual_batch_size <= 0:
                continue

            batch_dict = {
                'state': states[start_idx:end_idx],
                'action': actions[start_idx:end_idx],
                'reward': rewards[start_idx:end_idx],
                'done': dones[start_idx:end_idx],
                'next_state': next_states[start_idx:end_idx],
            }

            # Add PER specific data if available
            if self.is_per:
                batch_dict['indices'] = indices[start_idx:end_idx]
                batch_dict['weights'] = weights[start_idx:end_idx]

            self.batched_data.append(batch_dict)

        # For diagnostics
        self.memory_usage = self._estimate_memory_usage()

    def _estimate_memory_usage(self):
        """Estimate memory usage of the dataset in MB"""
        if not self.batched_data:
            return 0

        sample_batch = self.batched_data[0]
        batch_size_bytes = sum(
            tensor.element_size() * tensor.nelement()
            for tensor in sample_batch.values()
            if torch.is_tensor(tensor)
        )
        total_bytes = batch_size_bytes * len(self.batched_data)
        return total_bytes / (1024 * 1024)  # Convert to MB

    def __len__(self) -> int:
        """Return the number of batches"""
        return len(self.batched_data)

    def __getitem__(self, idx: int) -> Dict:
        """Return an entire pre-prepared batch"""
        return self.batched_data[idx]


class BatchedRLDataModule:
    """Data module with optimized batch-based sampling"""

    def __init__(self,
                 buffer,
                 dataset_sample_size=1000,
                 batch_size=32,
                 num_workers=0):
        self.buffer = buffer
        self.dataset_sample_size = dataset_sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_dataloader(self) -> DataLoader:
        """Create an optimized dataloader with pre-batched data"""
        dataset = BatchedRLDataset(
            self.buffer,
            sample_size=self.dataset_sample_size,
            batch_size=self.batch_size
        )

        # Log dataset information
        print(f"Created dataset with {len(dataset)} batches")
        print(f"Estimated memory usage: {dataset.memory_usage:.2f} MB")

        # Create a dataloader that returns one pre-made batch at a time
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,  # Each item is already a batch
            shuffle=False,  # Already randomly sampled
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=lambda x: x[0]  # Extract the batch from the list
        )

        return dataloader