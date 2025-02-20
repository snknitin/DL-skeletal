import torch
import time
from torch.utils.data import DataLoader
import numpy as np

# Import original and optimized implementations
from torch.utils.data import IterableDataset
from optimized_rl_dataset import RLMapDataset


# Original implementation (for comparison)
class OriginalRLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size
        self.is_per = isinstance(buffer, MockPrioritizedReplayBuffer)

    def __iter__(self):
        if self.is_per:
            batch, indices, weights = self.buffer.sample(self.sample_size)
            states, actions, rewards, dones, next_states = batch
            for i in range(len(dones)):
                yield states[i], actions[i], rewards[i], dones[i], next_states[i], indices[i], weights[i]
        else:
            states, actions, rewards, dones, next_states = self.buffer.sample(self.sample_size)
            for i in range(len(dones)):
                yield states[i], actions[i], rewards[i], dones[i], next_states[i]


# Mock ReplayBuffer for testing
class MockReplayBuffer:
    def __init__(self, buffer_size=10000, state_dim=4, action_dim=2):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

    def sample(self, batch_size):
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        next_states = torch.randn(batch_size, self.state_dim)
        return states, actions, rewards, dones, next_states


# Mock PrioritizedReplayBuffer for testing
class MockPrioritizedReplayBuffer(MockReplayBuffer):
    def sample(self, batch_size):
        base_sample = super().sample(batch_size)
        indices = torch.arange(batch_size)
        weights = torch.ones(batch_size)
        return base_sample, indices, weights


# Test function to compare performance
def test_dataset_performance(dataset_cls, buffer, sample_size=1000, batch_size=32, num_workers=0):
    # Create dataset
    start_time = time.time()
    dataset = dataset_cls(buffer, sample_size)
    dataset_creation_time = time.time() - start_time

    # Test dataloader
    start_time = time.time()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Iterate through dataloader
    batch_count = 0
    sample_count = 0
    for batch in dataloader:
        batch_count += 1
        if isinstance(batch, dict):
            sample_count += len(batch['state'])
        else:
            sample_count += len(batch[0])

    iteration_time = time.time() - start_time

    return {
        'dataset_creation_time': dataset_creation_time,
        'iteration_time': iteration_time,
        'batch_count': batch_count,
        'sample_count': sample_count
    }


# Run comparison tests
if __name__ == "__main__":



    # Create test buffers
    regular_buffer = MockReplayBuffer()
    prioritized_buffer = MockPrioritizedReplayBuffer()

    # Test configurations
    configs = [
        {'workers': 0, 'buffer_type': 'regular'},
        {'workers': 2, 'buffer_type': 'regular'},
        {'workers': 4, 'buffer_type': 'regular'},
        {'workers': 8, 'buffer_type': 'regular'},
        {'workers': 0, 'buffer_type': 'prioritized'},
        {'workers': 2, 'buffer_type': 'prioritized'},
    ]

    print("=== Performance Comparison ===")
    for config in configs:
        buffer = prioritized_buffer if config['buffer_type'] == 'prioritized' else regular_buffer
        workers = config['workers']

        print(f"\nTesting with {config['buffer_type']} buffer and {workers} workers:")

        # Test original implementation
        orig_results = test_dataset_performance(
            OriginalRLDataset, buffer, sample_size=4096, batch_size=256, num_workers=workers
        )

        # Test optimized implementation
        opt_results = test_dataset_performance(
            RLMapDataset, buffer, sample_size=4096, batch_size=256, num_workers=workers
        )

        # Print comparison
        print(f"  Original: Creation {orig_results['dataset_creation_time']:.4f}s, "
              f"Iteration {orig_results['iteration_time']:.4f}s, "
              f"Samples: {orig_results['sample_count']}")

        print(f"  Optimized: Creation {opt_results['dataset_creation_time']:.4f}s, "
              f"Iteration {opt_results['iteration_time']:.4f}s, "
              f"Samples: {opt_results['sample_count']}")

        speedup = orig_results['iteration_time'] / opt_results['iteration_time']
        print(f"  Speedup: {speedup:.2f}x faster iteration")