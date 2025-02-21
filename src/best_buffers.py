import torch
import time
from torch.utils.data import DataLoader, IterableDataset, Dataset
import numpy as np


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


# 1. Original Iterable Dataset implementation
class OriginalRLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

        # Identify buffer type
        test_sample = buffer.sample(1)
        self.is_per = isinstance(test_sample, tuple) and len(test_sample) == 3

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

    def get_dataloader(self, batch_size=32, num_workers=0):
        """Get dataloader specific to this dataset type"""
        # Deliberately allow setting num_workers to demonstrate the issue
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=0,  # Allow this for demonstration
            pin_memory=torch.cuda.is_available()
        )


# 2. Map-style implementation with individual access
class RLMapDataset(Dataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

        # Identify buffer type
        test_sample = buffer.sample(1)
        self.is_per = isinstance(test_sample, tuple) and len(test_sample) == 3

        # Sample the data once upfront
        self._sample_data()

    def _sample_data(self):
        """Sample data from buffer once and store it"""
        sample_result = self.buffer.sample(self.sample_size)

        if self.is_per:
            # PrioritizedReplayBuffer returns (batch, indices, weights)
            batch, self.indices, self.weights = sample_result
            self.states, self.actions, self.rewards, self.dones, self.next_states = batch
        else:
            # Regular ReplayBuffer returns (states, actions, rewards, dones, next_states) directly
            self.states, self.actions, self.rewards, self.dones, self.next_states = sample_result
            self.indices = None
            self.weights = None

    def __len__(self) -> int:
        """Return the number of samples"""
        return self.sample_size

    def __getitem__(self, idx: int):
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

    def get_dataloader(self, batch_size=32, num_workers=0):
        """Get dataloader specific to this dataset type"""
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=False,  # Already randomly sampled
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )


# Define a standalone collate function outside any class
def batch_collate_fn(batch_list):
    """Extract the single batch from a list containing one batch"""
    return batch_list[0]


# 3. Batch-optimized implementation
class BatchedRLDataset(Dataset):
    """Dataset that pre-organizes buffer samples into batches for efficient access"""

    def __init__(self, buffer, sample_size=200, batch_size=32):
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

    def __len__(self) -> int:
        """Return the number of batches"""
        return len(self.batched_data)

    def __getitem__(self, idx: int):
        """Return an entire pre-prepared batch"""
        return self.batched_data[idx]

    def get_dataloader(self, batch_size=None, num_workers=0):
        """Get dataloader specific to this dataset type"""
        return DataLoader(
            dataset=self,
            batch_size=1,  # Each dataset item is already a batch
            shuffle=False,  # Already randomly sampled
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            collate_fn=batch_collate_fn  # Use the standalone function
        )


# Test function to compare performance
def test_dataset_performance(dataset_cls, buffer, sample_size=1000, batch_size=32, num_workers=0):
    start_time = time.time()

    # Create dataset with appropriate parameters
    if dataset_cls == BatchedRLDataset:
        dataset = dataset_cls(buffer, sample_size=sample_size, batch_size=batch_size)
    else:
        dataset = dataset_cls(buffer, sample_size=sample_size)

    dataset_creation_time = time.time() - start_time

    # Get the appropriate dataloader for this dataset type
    start_time = time.time()
    dataloader = dataset.get_dataloader(batch_size=batch_size, num_workers=num_workers)

    # Iterate through dataloader and process batches
    batch_count = 0
    sample_count = 0
    expected_sample_count = sample_size
    processed_tensors = []

    for batch in dataloader:
        batch_count += 1

        # Count samples based on batch format
        if isinstance(batch, dict):
            # Map-style or batched dataset returns dict
            current_batch_size = len(batch['state'])
            # Keep reference to tensor to ensure it's computed
            processed_tensors.append(
                torch.matmul(batch['state'], torch.ones(batch['state'].shape[1], 1))
            )
        elif isinstance(batch, (tuple, list)):
            # Iterable dataset or other format
            if hasattr(batch[0], 'shape'):
                current_batch_size = batch[0].shape[0]
                # Keep reference to tensor to ensure it's computed
                processed_tensors.append(
                    torch.matmul(batch[0], torch.ones(batch[0].shape[1], 1))
                )
            else:
                # Fallback
                try:
                    current_batch_size = len(batch[0])
                except (TypeError, IndexError):
                    current_batch_size = 0
        else:
            # Unknown format - try to estimate
            try:
                current_batch_size = len(batch)
            except (TypeError, AttributeError):
                current_batch_size = 1  # Assume at least one sample

        sample_count += current_batch_size

    # Force computation of tensors
    for t in processed_tensors:
        _ = t.sum().item()

    iteration_time = time.time() - start_time

    # Calculate metrics
    batches_per_second = batch_count / iteration_time if iteration_time > 0 else 0
    samples_per_second = sample_count / iteration_time if iteration_time > 0 else 0
    duplication_factor = sample_count / expected_sample_count if expected_sample_count > 0 else 1.0

    return {
        'dataset_creation_time': dataset_creation_time,
        'iteration_time': iteration_time,
        'batch_count': batch_count,
        'sample_count': sample_count,
        'expected_sample_count': expected_sample_count,
        'duplication_factor': duplication_factor,
        'batches_per_second': batches_per_second,
        'samples_per_second': samples_per_second
    }


# Run comparison tests
if __name__ == "__main__":
    # Create test buffers
    regular_buffer = MockReplayBuffer()
    prioritized_buffer = MockPrioritizedReplayBuffer()

    # Test parameters
    SAMPLE_SIZE = 10000  # Larger dataset for more realistic testing
    BATCH_SIZE = 256

    # Test configurations for thorough comparison
    configs = [
        {'workers': 0, 'buffer_type': 'regular', 'name': 'Single-threaded, Regular Buffer'},
        {'workers': 0, 'buffer_type': 'prioritized', 'name': 'Single-threaded, Prioritized Buffer'},
        {'workers': 4, 'buffer_type': 'regular', 'name': 'Multi-threaded (4), Regular Buffer'},
        {'workers': 4, 'buffer_type': 'prioritized', 'name': 'Multi-threaded (4), Prioritized Buffer'},
    ]

    # Store results for comparison
    results = {}

    print("======= RL Dataset Performance Comparison =======")
    print(f"Sample size: {SAMPLE_SIZE}, Batch size: {BATCH_SIZE}")
    print("================================================\n")

    # Run all tests
    for config in configs:
        buffer = prioritized_buffer if config['buffer_type'] == 'prioritized' else regular_buffer
        workers = config['workers']
        config_name = config['name']

        print(f"\n--- Testing: {config_name} ---")

        # Allow OriginalRLDataset to use workers to demonstrate the issue
        orig_results = test_dataset_performance(
            OriginalRLDataset, buffer, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, num_workers=workers
        )

        # Test map dataset implementation
        map_results = test_dataset_performance(
            RLMapDataset, buffer, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, num_workers=workers
        )

        # Test batched dataset implementation
        batched_results = test_dataset_performance(
            BatchedRLDataset, buffer, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, num_workers=workers
        )

        # Store results
        results[config_name] = {
            'original': orig_results,
            'map': map_results,
            'batched': batched_results
        }

        # Print detailed results for this configuration
        print(f"\nResults for {config_name}:")
        print(
            f"{'Dataset Type':<15} {'Creation(s)':<10} {'Iter(s)':<10} {'Batches':<8} {'Samples':<10} {'Expected':<10} {'Duplication':<10} {'Batch/s':<8} {'Sample/s':<10}")
        print("-" * 100)

        print(f"{'Original':<15} {orig_results['dataset_creation_time']:<10.4f} "
              f"{orig_results['iteration_time']:<10.4f} {orig_results['batch_count']:<8} "
              f"{orig_results['sample_count']:<10} {orig_results['expected_sample_count']:<10} "
              f"{orig_results['duplication_factor']:<10.2f} {orig_results['batches_per_second']:<8.1f} "
              f"{orig_results['samples_per_second']:<10.1f}")

        print(f"{'Map Dataset':<15} {map_results['dataset_creation_time']:<10.4f} "
              f"{map_results['iteration_time']:<10.4f} {map_results['batch_count']:<8} "
              f"{map_results['sample_count']:<10} {map_results['expected_sample_count']:<10} "
              f"{map_results['duplication_factor']:<10.2f} {map_results['batches_per_second']:<8.1f} "
              f"{map_results['samples_per_second']:<10.1f}")

        print(f"{'Batched':<15} {batched_results['dataset_creation_time']:<10.4f} "
              f"{batched_results['iteration_time']:<10.4f} {batched_results['batch_count']:<8} "
              f"{batched_results['sample_count']:<10} {batched_results['expected_sample_count']:<10} "
              f"{batched_results['duplication_factor']:<10.2f} {batched_results['batches_per_second']:<8.1f} "
              f"{batched_results['samples_per_second']:<10.1f}")

        # Highlight duplication issue
        if orig_results['duplication_factor'] > 1.1:
            print(
                f"\n⚠️ WARNING: IterableDataset duplicated data {orig_results['duplication_factor']:.1f}x with {workers} workers!")
            print(f"   This means each sample was processed multiple times, wasting computation.")

        # Calculate speedups
        map_speedup = orig_results['iteration_time'] / map_results['iteration_time'] if map_results[
                                                                                            'iteration_time'] > 0 else 0
        batched_speedup = orig_results['iteration_time'] / batched_results['iteration_time'] if batched_results[
                                                                                                    'iteration_time'] > 0 else 0
        batched_vs_map = map_results['iteration_time'] / batched_results['iteration_time'] if batched_results[
                                                                                                  'iteration_time'] > 0 else 0

        print(f"\nSpeedup factors (iteration time):")
        print(f"- Map vs Original: {map_speedup:.2f}x faster")
        print(f"- Batched vs Original: {batched_speedup:.2f}x faster")
        print(f"- Batched vs Map: {batched_vs_map:.2f}x faster")

    # Print summary across all configurations
    print("\n\n========== SUMMARY ==========")
    print("Average speedup factors across all configurations:")

    avg_map_speedup = sum(results[cfg]['original']['iteration_time'] / results[cfg]['map']['iteration_time']
                          if results[cfg]['map']['iteration_time'] > 0 else 0
                          for cfg in results) / len(results)

    avg_batched_speedup = sum(results[cfg]['original']['iteration_time'] / results[cfg]['batched']['iteration_time']
                              if results[cfg]['batched']['iteration_time'] > 0 else 0
                              for cfg in results) / len(results)

    avg_batched_vs_map = sum(results[cfg]['map']['iteration_time'] / results[cfg]['batched']['iteration_time']
                             if results[cfg]['batched']['iteration_time'] > 0 else 0
                             for cfg in results) / len(results)

    print(f"- Map vs Original: {avg_map_speedup:.2f}x faster")
    print(f"- Batched vs Original: {avg_batched_speedup:.2f}x faster")
    print(f"- Batched vs Map: {avg_batched_vs_map:.2f}x faster")

    # Print key insights
    print("\n========== KEY INSIGHTS ==========")

    # Check for duplication in multi-worker setups
    multi_worker_configs = [cfg for cfg in configs if cfg['workers'] > 0]
    if multi_worker_configs:
        avg_orig_duplication = sum(results[cfg['name']]['original']['duplication_factor']
                                   for cfg in multi_worker_configs) / len(multi_worker_configs)

        if avg_orig_duplication > 1.1:
            print(
                f"⚠️ IterableDataset with multiple workers causes data duplication ({avg_orig_duplication:.1f}x on average)")
            print("   This can waste computation and lead to incorrect training dynamics")
            print("   Recommendation: Use RLMapDataset or BatchedRLDataset with multiple workers instead")

    # Identify fastest implementation
    fastest_dataset = None
    max_speedup = 0

    for cfg in configs:
        map_speed = results[cfg['name']]['map']['samples_per_second']
        batched_speed = results[cfg['name']]['batched']['samples_per_second']

        if map_speed > max_speedup:
            max_speedup = map_speed
            fastest_dataset = 'map'

        if batched_speed > max_speedup:
            max_speedup = batched_speed
            fastest_dataset = 'batched'

    print(f"🏆 The {fastest_dataset} dataset implementation achieved the highest throughput")
    print(f"   Peak performance: {max_speedup:.1f} samples per second")