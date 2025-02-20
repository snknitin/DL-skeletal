# dataloader_tester.py
import torch
import lightning as pl
from torch.utils.data import DataLoader, IterableDataset
from typing import Union
import numpy as np
import time

# Simplified version of your RLDataset
class TestRLDataset(IterableDataset):
    def __init__(self, sample_size: int = 4096):
        self.sample_size = sample_size

    def __iter__(self):
        # Generate dummy data of specified size
        for i in range(self.sample_size):
            # Simulate your experience tuple with smaller tensors
            state = torch.randn(4)
            action = torch.tensor([1])
            reward = torch.tensor([0.5])
            done = torch.tensor([0])
            next_state = torch.randn(4)
            yield state, action, reward, done, next_state


class TestRLDataset(IterableDataset):
    def __init__(self, sample_size: int = 4096):
        self.sample_size = sample_size

    def __iter__(self):
        # Generate dummy data of specified size
        for i in range(self.sample_size):
            # Simulate your experience tuple with smaller tensors
            state = torch.randn(4)
            action = torch.tensor([1])
            reward = torch.tensor([0.5])
            done = torch.tensor([0])
            next_state = torch.randn(4)
            yield state, action, reward, done, next_state


class TestLightningModule(pl.LightningModule):
    def __init__(self, dataset_sample_size: int = 4096, batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()

        # Simple network for testing
        self.net = torch.nn.Linear(4, 1)

        # Counters for monitoring
        self.batches_in_epoch = 0
        self.samples_in_batch = 0
        self.current_epoch_steps = 0

    def _dataloader(self) -> DataLoader:
        dataset = TestRLDataset(sample_size=self.hparams.dataset_sample_size)
        dataloader =  DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        # print("\n\nLenght of dataloader :", len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self._dataloader()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, next_states = batch

        # Print batch information
        self.samples_in_batch = states.shape[0]
        self.batches_in_epoch += 1
        self.current_epoch_steps += 1

        if self.global_step%100==0:

            print(f"\nStep Information:")
            print(f"Global Step: {self.global_step}")
            print(f"Current Epoch: {self.current_epoch}")
            print(f"Batch Index: {batch_idx}")
            print(f"Samples in Batch: {self.samples_in_batch}")
            print(f"Batches in Epoch so far: {self.batches_in_epoch}")
            print(f"Steps in Epoch so far: {self.current_epoch_steps}")

        # Dummy loss computation
        loss = self.net(states).mean()
        return loss

    def on_train_epoch_end(self):
        print(f"\nEpoch {self.current_epoch} Summary:")
        print(f"Total batches processed: {self.batches_in_epoch}")
        print(f"Total steps in epoch: {self.current_epoch_steps}")
        print("-" * 50)

        # Reset counters for next epoch
        self.batches_in_epoch = 0
        self.current_epoch_steps = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def test_dataloader_behavior(
        dataset_size: int = 4096,
        batch_size: int = 256,
        num_workers: int = 4,
        max_steps: int = 200
):
    model = TestLightningModule(
        dataset_sample_size=dataset_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )

    print(f"\nTesting with configuration:")
    print(f"Dataset Size: {dataset_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Workers: {num_workers}")
    print(f"Max Steps: {max_steps}")
    print("-" * 50)

    trainer.fit(model)


if __name__ == "__main__":
    # Test different configurations
    configs = [

        # {"dataset_size": 256, "batch_size": 256, "num_workers": 1},
        # {"dataset_size": 4096, "batch_size": 256, "num_workers": 2},
        # {"dataset_size": 4096, "batch_size": 256, "num_workers": 4},
        {"dataset_size": 4096, "batch_size": 256, "num_workers": 8},
    ]

    for config in configs:
        print("\n\n\n\n")
        test_dataloader_behavior(**config)