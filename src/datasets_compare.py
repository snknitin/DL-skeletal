import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
import time
import numpy as np
from argparse import ArgumentParser
import os
from datetime import datetime

# Import dataset classes - these should be in your existing modules
from best_buffers import OriginalRLDataset, RLMapDataset, BatchedRLDataset
from best_buffers import MockReplayBuffer, MockPrioritizedReplayBuffer


# ==========================
# Lightning Modules
# ==========================

class SimpleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RLBaseModule(pl.LightningModule):
    def __init__(
            self,
            buffer,
            state_dim=4,
            action_dim=2,
            learning_rate=0.001,
            gamma=0.99,
            dataset_sample_size=1000,
            batch_size=64,
            num_workers=0,
            dataset_type='original'  # 'original', 'map', or 'batched'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['buffer'])
        self.buffer = buffer

        # Initialize networks
        self.q_net = SimpleQNetwork(state_dim, action_dim)
        self.target_net = SimpleQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Track metrics
        self.total_train_time = 0
        self.batch_times = []
        self.loss_values = []
        self.batch_sizes = []
        self.total_samples_processed = 0

        # Create dataset on initialization
        self.dataset = self.get_dataset()

    def get_dataset(self):
        """Create dataset based on specified type"""
        if self.hparams.dataset_type == 'original':
            return OriginalRLDataset(
                self.buffer,
                sample_size=self.hparams.dataset_sample_size
            )
        elif self.hparams.dataset_type == 'map':
            return RLMapDataset(
                self.buffer,
                sample_size=self.hparams.dataset_sample_size
            )
        elif self.hparams.dataset_type == 'batched':
            return BatchedRLDataset(
                self.buffer,
                sample_size=self.hparams.dataset_sample_size,
                batch_size=self.hparams.batch_size
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")

    def train_dataloader(self):
        return self.dataset.get_dataloader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def _compute_loss(self, batch):
        """Compute Q-learning loss"""
        # Unpack batch based on format
        if isinstance(batch, dict):
            # Map-style or batched dataset returns dict
            states = batch['state']
            actions = batch['action']
            rewards = batch['reward']
            dones = batch['done']
            next_states = batch['next_state']
        elif isinstance(batch, (tuple, list)):
            # Iterable dataset returns tuple
            if len(batch) >= 5:
                states, actions, rewards, dones, next_states = batch[:5]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)} with len {len(batch)}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # Track batch size
        if hasattr(states, 'shape'):
            batch_size = states.shape[0]
        else:
            batch_size = len(states)
        self.batch_sizes.append(batch_size)
        self.total_samples_processed += batch_size

        # Get current Q values
        current_q = self.q_net(states)

        # Select Q values for taken actions
        # For simplicity in this benchmark, we'll use the first action dimension as an index
        action_idx = torch.argmax(actions, dim=1, keepdim=True)
        selected_q = current_q.gather(1, action_idx.long())

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.hparams.gamma * max_next_q

        # Compute loss
        loss = F.mse_loss(selected_q, target_q)
        return loss

    def training_step(self, batch, batch_idx):
        # Measure batch processing time
        start_time = time.time()

        # Compute loss
        try:
            loss = self._compute_loss(batch)
        except Exception as e:
            self.log('error', 1.0, prog_bar=True)
            print(f"Error processing batch: {e}")
            print(f"Batch type: {type(batch)}")
            if isinstance(batch, (tuple, list)):
                print(f"Batch length: {len(batch)}")
                for i, item in enumerate(batch[:min(5, len(batch))]):
                    print(f"Item {i} type: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"Item {i} shape: {item.shape}")
            raise

        # Record metrics
        batch_time = time.time() - start_time
        self.batch_times.append(batch_time)
        self.loss_values.append(loss.item())

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('batch_time', batch_time, prog_bar=True)

        # Return loss for backprop
        return loss

    def on_train_epoch_end(self):
        """Log epoch statistics"""
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            avg_loss = np.mean(self.loss_values)
            avg_batch_size = np.mean(self.batch_sizes)

            # Log epoch metrics
            self.log('epoch_avg_batch_time', avg_batch_time)
            self.log('epoch_avg_loss', avg_loss)
            self.log('epoch_avg_batch_size', avg_batch_size)
            self.log('total_samples_processed', float(self.total_samples_processed))

    def configure_optimizers(self):
        return torch.optim.Adam(self.q_net.parameters(), lr=self.hparams.learning_rate)


# ==========================
# Benchmark Runner
# ==========================

class BenchmarkRunner:
    def __init__(self,
                 benchmark_name="RL Dataset Benchmark",
                 state_dim=4,
                 action_dim=2,
                 buffer_size=100000,
                 dataset_sample_size=10000,
                 batch_size=128,
                 num_workers=2,
                 num_epochs=3,
                 use_prioritized_buffer=False,
                 accelerator='cpu',
                 devices=None):
        self.benchmark_name = benchmark_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.dataset_sample_size = dataset_sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.use_prioritized_buffer = use_prioritized_buffer
        self.accelerator = accelerator
        self.devices = devices

        # Create buffer
        if use_prioritized_buffer:
            self.buffer = MockPrioritizedReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim
            )
        else:
            self.buffer = MockReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim
            )

        # Progress bar callback
        self.progress_bar = TQDMProgressBar(refresh_rate=10)

    def _create_trainer(self, max_epochs):
        """Create Lightning Trainer"""
        return pl.Trainer(
            max_epochs=max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=[self.progress_bar],
            enable_progress_bar=True,
            logger=False,  # Disable logging to avoid file creation
            enable_checkpointing=False,  # Disable checkpointing
        )

    def run_benchmark(self):
        """Run benchmarks for all dataset types"""
        results = {}
        dataset_types = ['original', 'map', 'batched']

        print(f"\n{'=' * 60}")
        print(f"  {self.benchmark_name}")
        print(f"  {'Buffer Type:':<20} {'Prioritized' if self.use_prioritized_buffer else 'Regular'}")
        print(f"  {'Accelerator:':<20} {self.accelerator}")
        if self.accelerator == 'gpu':
            device_info = torch.cuda.get_device_name(self.devices[0]) if self.devices else torch.cuda.get_device_name(0)
            print(f"  {'Device:':<20} {device_info}")
        print(f"  {'Sample Size:':<20} {self.dataset_sample_size}")
        print(f"  {'Batch Size:':<20} {self.batch_size}")
        print(f"  {'Workers:':<20} {self.num_workers}")
        print(f"  {'Epochs:':<20} {self.num_epochs}")
        print(f"{'=' * 60}\n")

        for dataset_type in dataset_types:
            actual_workers = 0 if dataset_type == 'original' else self.num_workers
            print(f"\nBenchmarking {dataset_type} dataset with {actual_workers} workers...")

            # Create model with current dataset type
            model = RLBaseModule(
                buffer=self.buffer,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                dataset_sample_size=self.dataset_sample_size,
                batch_size=self.batch_size,
                num_workers=actual_workers,
                dataset_type=dataset_type
            )

            # Create trainer
            trainer = self._create_trainer(max_epochs=self.num_epochs)

            # Run training
            start_time = time.time()
            trainer.fit(model)
            total_time = time.time() - start_time

            # Calculate metrics
            avg_epoch_time = total_time / self.num_epochs
            batches_per_epoch = len(model.batch_times) / self.num_epochs if model.batch_times else 0
            avg_batch_time = np.mean(model.batch_times) if model.batch_times else 0
            batches_per_second = 1 / avg_batch_time if avg_batch_time > 0 else 0
            samples_per_second = model.total_samples_processed / total_time if total_time > 0 else 0
            avg_batch_size = np.mean(model.batch_sizes) if model.batch_sizes else 0

            # Store results
            results[dataset_type] = {
                'total_time': total_time,
                'avg_epoch_time': avg_epoch_time,
                'avg_batch_time': avg_batch_time,
                'batches_per_epoch': batches_per_epoch,
                'batches_per_second': batches_per_second,
                'samples_per_second': samples_per_second,
                'total_samples': model.total_samples_processed,
                'avg_batch_size': avg_batch_size,
                'avg_loss': np.mean(model.loss_values) if model.loss_values else 0
            }

            # Print immediate results
            print(f"  ✓ Total time: {total_time:.2f}s")
            print(f"  ✓ Avg epoch time: {avg_epoch_time:.2f}s")
            print(f"  ✓ Batches/second: {batches_per_second:.2f}")
            print(f"  ✓ Samples/second: {samples_per_second:.2f}")

            # Clear GPU memory between runs
            if self.accelerator == 'gpu':
                torch.cuda.empty_cache()

        # Print comparative results
        self._print_comparison(results)
        return results

    def _print_comparison(self, results):
        """Print comparative benchmark results"""
        if not results:
            return

        # Get baseline (original) results
        baseline_time = results['original']['total_time']

        print("\n\n" + "=" * 70)
        print(f"  BENCHMARK RESULTS SUMMARY - {self.accelerator.upper()}")
        print("=" * 70)

        # Table header
        print(
            f"\n{'Dataset Type':<15} {'Total Time':<12} {'Speedup':<10} {'Batches/sec':<12} {'Samples/sec':<12} {'Avg Loss':<10}")
        print("-" * 75)

        # Table rows
        for dataset_type, metrics in results.items():
            speedup = baseline_time / metrics['total_time'] if metrics['total_time'] > 0 else 0
            print(f"{dataset_type:<15} {metrics['total_time']:<12.2f}s {speedup:<10.2f}x "
                  f"{metrics['batches_per_second']:<12.2f} {metrics['samples_per_second']:<12.2f} "
                  f"{metrics['avg_loss']:<10.4f}")

        # Find fastest dataset type
        fastest = min(results.items(), key=lambda x: x[1]['total_time'])
        highest_throughput = max(results.items(), key=lambda x: x[1]['samples_per_second'])

        print(f"\n🏆 Fastest implementation: {fastest[0]} dataset")
        print(f"🚀 Highest throughput: {highest_throughput[0]} dataset "
              f"({highest_throughput[1]['samples_per_second']:.2f} samples/sec)")

        # Print hardware-specific insights
        if self.accelerator == 'gpu':
            print("\nGPU-SPECIFIC INSIGHTS:")
            if self.num_workers > 0 and results['map']['samples_per_second'] > results['original'][
                'samples_per_second']:
                worker_speedup = results['map']['samples_per_second'] / results['original']['samples_per_second']
                print(
                    f"✓ Multi-worker dataloading ({self.num_workers} workers) provides {worker_speedup:.1f}x throughput")

            if results['batched']['samples_per_second'] > results['map']['samples_per_second']:
                print("✓ Pre-batched data significantly reduces GPU data loading overhead")
                print("✓ This suggests your GPU compute is being underutilized with standard loading")
            else:
                print("✓ Standard map dataset performs well, suggesting good GPU utilization")

        else:  # CPU insights
            print("\nCPU-SPECIFIC INSIGHTS:")
            if self.num_workers > 1:
                if results['map']['samples_per_second'] > 1.3 * results['original']['samples_per_second']:
                    print(f"✓ Multi-worker dataloading is beneficial on CPU with {self.num_workers} workers")
                else:
                    print("✓ Multi-worker loading shows limited benefit - CPU may be the bottleneck")

            if results['batched']['avg_batch_time'] < results['map']['avg_batch_time']:
                print("✓ Batched data preparation reduces processing overhead on CPU")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        if fastest[0] == 'batched':
            print("✓ The BatchedRLDataset provides the best performance overall")
            print("✓ Consider optimizing batch size for memory vs speed tradeoff")
        elif fastest[0] == 'map':
            print("✓ The RLMapDataset provides good performance with simpler implementation")

        # Worker recommendations
        if self.num_workers > 0 and self.accelerator == 'gpu':
            optimal_workers = "multiple" if results['map']['samples_per_second'] > 1.3 * results['original'][
                'samples_per_second'] else "fewer"
            print(f"✓ For GPU training, {optimal_workers} workers is optimal")


# ==========================
# Main Benchmark Function
# ==========================

def run_all_benchmarks(args):
    # Setup accelerator configurations to test
    configs = []
    args.gpu = True  # Comment this if gpu is unavailable
    # CPU configurations
    cpu_workers = [0, 4]  # Test single-threaded and multi-threaded
    for workers in cpu_workers:
        configs.append({
            'name': f"CPU-{workers}workers",
            'accelerator': 'cpu',
            'devices': 1,
            'num_workers': workers
        })

    # GPU configurations if available
    if torch.cuda.is_available() and args.gpu:
        gpu_workers = [0, 2, 4]  # Test different worker counts on GPU
        for workers in gpu_workers:
            configs.append({
                'name': f"GPU-{workers}workers",
                'accelerator': 'gpu',
                'devices': [0],  # Use first GPU
                'num_workers': workers
            })

    # Store all results
    all_results = {}

    # Run benchmarks for each configuration
    for config in configs:
        print(f"\n\n{'*' * 80}")
        print(f"RUNNING BENCHMARK: {config['name']}")
        print(f"{'*' * 80}")

        # Regular buffer benchmark
        regular_benchmark = BenchmarkRunner(
            benchmark_name=f"Regular Buffer - {config['name']}",
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            buffer_size=args.buffer_size,
            dataset_sample_size=args.sample_size,
            batch_size=args.batch_size,
            num_workers=config['num_workers'],
            num_epochs=args.epochs,
            use_prioritized_buffer=False,
            accelerator=config['accelerator'],
            devices=config['devices']
        )
        regular_results = regular_benchmark.run_benchmark()

        # Prioritized buffer benchmark
        prioritized_benchmark = BenchmarkRunner(
            benchmark_name=f"Prioritized Buffer - {config['name']}",
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            buffer_size=args.buffer_size,
            dataset_sample_size=args.sample_size,
            batch_size=args.batch_size,
            num_workers=config['num_workers'],
            num_epochs=args.epochs,
            use_prioritized_buffer=True,
            accelerator=config['accelerator'],
            devices=config['devices']
        )
        prioritized_results = prioritized_benchmark.run_benchmark()

        # Store results
        all_results[config['name']] = {
            'regular': regular_results,
            'prioritized': prioritized_results,
            'config': config
        }

    # Final cross-configuration analysis
    print_final_analysis(all_results)

    return all_results


def print_final_analysis(all_results):
    print("\n\n" + "=" * 100)
    print(f"  COMPREHENSIVE ANALYSIS ACROSS ALL CONFIGURATIONS")
    print("=" * 100)

    # Find best configurations for each dataset type
    dataset_types = ['original', 'map', 'batched']

    best_configs = {}
    best_throughputs = {}

    for dataset_type in dataset_types:
        best_throughput = 0
        best_config = None
        best_buffer_type = None

        for config_name, results in all_results.items():
            # Check regular buffer
            if results['regular'][dataset_type]['samples_per_second'] > best_throughput:
                best_throughput = results['regular'][dataset_type]['samples_per_second']
                best_config = config_name
                best_buffer_type = 'regular'

            # Check prioritized buffer
            if results['prioritized'][dataset_type]['samples_per_second'] > best_throughput:
                best_throughput = results['prioritized'][dataset_type]['samples_per_second']
                best_config = config_name
                best_buffer_type = 'prioritized'

        best_configs[dataset_type] = (best_config, best_buffer_type)
        best_throughputs[dataset_type] = best_throughput

    # Print best configurations
    print("\nBest Configuration For Each Dataset Type:")
    print(f"{'Dataset Type':<15} {'Configuration':<20} {'Buffer Type':<15} {'Throughput':<15}")
    print("-" * 70)

    for dataset_type in dataset_types:
        best_config, best_buffer = best_configs[dataset_type]
        throughput = best_throughputs[dataset_type]
        print(f"{dataset_type:<15} {best_config:<20} {best_buffer:<15} {throughput:<15.2f} samples/sec")

    # Find overall best dataset/configuration combo
    overall_best_dataset = max(best_throughputs.items(), key=lambda x: x[1])[0]
    best_config, best_buffer = best_configs[overall_best_dataset]
    best_perf = best_throughputs[overall_best_dataset]

    print(f"\n🏆 OVERALL BEST COMBINATION:")
    print(f"  • Dataset: {overall_best_dataset}")
    print(f"  • Configuration: {best_config}")
    print(f"  • Buffer Type: {best_buffer}")
    print(f"  • Throughput: {best_perf:.2f} samples/sec")

    # GPU vs CPU analysis
    gpu_configs = [cfg for cfg in all_results.keys() if 'GPU' in cfg]
    cpu_configs = [cfg for cfg in all_results.keys() if 'CPU' in cfg]

    if gpu_configs and cpu_configs:
        # Find best GPU and CPU performance
        best_gpu_throughput = 0
        best_cpu_throughput = 0

        for cfg in gpu_configs:
            for buffer_type in ['regular', 'prioritized']:
                for dataset_type in dataset_types:
                    throughput = all_results[cfg][buffer_type][dataset_type]['samples_per_second']
                    best_gpu_throughput = max(best_gpu_throughput, throughput)

        for cfg in cpu_configs:
            for buffer_type in ['regular', 'prioritized']:
                for dataset_type in dataset_types:
                    throughput = all_results[cfg][buffer_type][dataset_type]['samples_per_second']
                    best_cpu_throughput = max(best_cpu_throughput, throughput)

        gpu_speedup = best_gpu_throughput / best_cpu_throughput if best_cpu_throughput > 0 else 0

        print(f"\n⚡ GPU vs CPU COMPARISON:")
        print(f"  • Best GPU throughput: {best_gpu_throughput:.2f} samples/sec")
        print(f"  • Best CPU throughput: {best_cpu_throughput:.2f} samples/sec")
        print(f"  • GPU speedup: {gpu_speedup:.2f}x faster")

    # Worker scaling analysis
    print("\n📊 WORKER SCALING ANALYSIS:")

    # For each dataset type, analyze worker scaling on GPU
    gpu_worker_configs = {}
    for cfg_name, results in all_results.items():
        if 'GPU' in cfg_name:
            worker_count = int(cfg_name.split('-')[1].replace('workers', ''))
            gpu_worker_configs[worker_count] = cfg_name

    if gpu_worker_configs:
        worker_counts = sorted(gpu_worker_configs.keys())

        for dataset_type in ['map', 'batched']:  # Skip original as it doesn't use workers
            print(f"\n{dataset_type.capitalize()} Dataset Worker Scaling (GPU):")
            print(f"{'Workers':<10} {'Samples/sec':<15} {'Scaling':<10}")
            print("-" * 40)

            baseline_throughput = None
            for workers in worker_counts:
                cfg_name = gpu_worker_configs[workers]
                throughput = all_results[cfg_name]['regular'][dataset_type]['samples_per_second']

                if baseline_throughput is None:
                    baseline_throughput = throughput
                    scaling = 1.0
                else:
                    scaling = throughput / baseline_throughput

                print(f"{workers:<10} {throughput:<15.2f} {scaling:<10.2f}x")

    # Key takeaways
    print("\n🔑 KEY TAKEAWAYS:")

    # Determine best dataset overall
    dataset_votes = {dt: 0 for dt in dataset_types}
    for config_name, results in all_results.items():
        for buffer_type in ['regular', 'prioritized']:
            best_dt = max(dataset_types,
                          key=lambda dt: results[buffer_type][dt]['samples_per_second'])
            dataset_votes[best_dt] += 1

    best_overall_dataset = max(dataset_votes.items(), key=lambda x: x[1])[0]

    print(f"1. The {best_overall_dataset} dataset implementation performs best in most scenarios")

    # Worker benefit analysis
    multi_worker_benefit = False
    if any('GPU' in cfg for cfg in all_results.keys()):
        for dataset_type in ['map', 'batched']:
            for cfg_name, results in all_results.items():
                if 'GPU' in cfg_name and '-0workers' not in cfg_name:
                    single_worker_cfg = cfg_name.replace(f"-{cfg_name.split('-')[1]}", "-0workers")
                    if single_worker_cfg in all_results:
                        multi_worker_throughput = results['regular'][dataset_type]['samples_per_second']
                        single_worker_throughput = all_results[single_worker_cfg]['regular'][dataset_type][
                            'samples_per_second']
                        if multi_worker_throughput > 1.2 * single_worker_throughput:
                            multi_worker_benefit = True
                            break

        if multi_worker_benefit:
            print(f"2. Using multiple workers significantly improves GPU throughput")
        else:
            print(f"2. Multiple workers show limited benefit on GPU - compute is likely the bottleneck")

    # PER overhead
    per_overhead = 0
    per_samples = 0
    for config_name, results in all_results.items():
        for dataset_type in dataset_types:
            reg_time = results['regular'][dataset_type]['total_time']
            pri_time = results['prioritized'][dataset_type]['total_time']
            if reg_time > 0:
                overhead = (pri_time - reg_time) / reg_time * 100
                per_overhead += overhead
                per_samples += 1

    avg_per_overhead = per_overhead / per_samples if per_samples > 0 else 0
    print(f"3. Prioritized Experience Replay adds {avg_per_overhead:.1f}% overhead on average")

    # Final recommendation
    print(f"\n📝 FINAL RECOMMENDATION:")
    print(f"Based on the comprehensive analysis, we recommend:")
    print(f"• Dataset: {best_overall_dataset}")
    if 'batched' in best_overall_dataset:
        print(f"• Pre-organize data into batch-sized chunks for optimal GPU throughput")
    print(f"• Workers: {2 if multi_worker_benefit else 0} for optimal GPU utilization")
    print(
        f"• Buffer: {'Either regular or prioritized (modest overhead)' if avg_per_overhead < 20 else 'Regular buffer for speed, prioritized if learning stability is critical'}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Comprehensive RL Dataset Benchmark")
    parser.add_argument("--state_dim", type=int, default=8, help="State dimension")
    parser.add_argument("--action_dim", type=int, default=4, help="Action dimension")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Buffer size")
    parser.add_argument("--sample_size", type=int, default=8192, help="Dataset sample size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true",  help="Test GPU configurations")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Run all benchmarks and get results
    results = run_all_benchmarks(args)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.pt")
    torch.save(results, result_file)
    print(f"\nResults saved to {result_file}")

    print("\nBenchmark complete! Use these insights to optimize your RL training pipeline.")