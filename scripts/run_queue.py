"""
Ablation experiment queue runner.

Launches each experiment as a subprocess so that torch.distributed process
groups are properly initialized and cleaned up between runs.

Usage — single GPU (or CPU):
    python scripts/run_queue.py

Usage — multi-GPU (torchrun):
    python scripts/run_queue.py --gpus 4

Usage — other options:
    python scripts/run_queue.py --queue config/experiments/queue.yaml
    python scripts/run_queue.py --gpus 4 --dry-run
    python scripts/run_queue.py --start 2          # skip first N experiments
    python scripts/run_queue.py --gpus 4 --node-rank 0 --nnodes 2 --master-addr host --master-port 29500

When --gpus > 1 the runner calls:
    torchrun --standalone --nproc_per_node=<gpus> main.py --config <config>

When --gpus == 1 (default) the runner calls:
    python main.py --config <config>

Each experiment runs in its own subprocess, so DDP process groups are fully
isolated between experiments.
"""
import sys
import argparse
import subprocess
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.toolkit import load_yaml


def get_args():
    parser = argparse.ArgumentParser(description='Run ablation experiment queue')
    parser.add_argument('--queue', '-q', type=str,
                        default='config/experiments/queue.yaml',
                        help='Path to queue YAML file')
    parser.add_argument('--gpus', '-g', type=int, default=1,
                        help='Number of GPUs per node (1 = no torchrun, >1 = torchrun)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running them')
    parser.add_argument('--start', type=int, default=0,
                        help='Skip first N experiments (0-indexed)')
    # Multi-node options (passed through to torchrun)
    parser.add_argument('--nnodes', type=int, default=1,
                        help='Number of nodes (for multi-node torchrun)')
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Rank of this node (for multi-node torchrun)')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1',
                        help='Master node address (for multi-node torchrun)')
    parser.add_argument('--master-port', type=int, default=29500,
                        help='Master node port (for multi-node torchrun)')
    return parser.parse_args()


def build_command(config_path: str, args) -> list[str]:
    """Build the subprocess command for one experiment."""
    main_py = str(Path(__file__).resolve().parent.parent / 'main.py')

    if args.gpus > 1 or args.nnodes > 1:
        cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            f'--nproc_per_node={args.gpus}',
            f'--nnodes={args.nnodes}',
            f'--node_rank={args.node_rank}',
            f'--master_addr={args.master_addr}',
            f'--master_port={args.master_port}',
            '--standalone' if args.nnodes == 1 else '',
        ]
        # remove empty strings (--standalone only for single-node)
        cmd = [c for c in cmd if c]
        cmd += [main_py, '--config', config_path]
    else:
        cmd = [sys.executable, main_py, '--config', config_path]

    return cmd


def run_experiment(cmd: list[str]) -> int:
    """Run subprocess, stream output, return exit code."""
    proc = subprocess.Popen(cmd, stdout=None, stderr=None)  # inherit parent stdio
    proc.wait()
    return proc.returncode


def main():
    args = get_args()

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"ERROR: Queue file not found: {queue_path}")
        sys.exit(1)

    queue = load_yaml(str(queue_path))
    experiments = queue.get('experiments', [])

    if not experiments:
        print("No experiments in queue.")
        return

    mode = f"torchrun --nproc_per_node={args.gpus}" if args.gpus > 1 else "python"
    print(f"Queue  : {queue_path}")
    print(f"Mode   : {mode}")
    print(f"Total  : {len(experiments)} experiments")
    if args.start > 0:
        print(f"Skipping first {args.start} experiment(s).")
    print()

    results = []

    for i, config_path in enumerate(experiments):
        status_prefix = f"[{i+1}/{len(experiments)}]"

        if i < args.start:
            print(f"{status_prefix} SKIP  {config_path}")
            results.append((config_path, 'skipped'))
            continue

        config_file = Path(config_path)
        if not config_file.exists():
            print(f"{status_prefix} ERROR config not found: {config_path}")
            results.append((config_path, 'config_not_found'))
            continue

        run_name = load_yaml(str(config_file)).get('logging', {}).get('wandb_run_name', config_path)
        cmd = build_command(config_path, args)

        print(f"{status_prefix} START {run_name}  ({config_path})")
        print(f"         CMD: {' '.join(cmd)}")

        if args.dry_run:
            results.append((config_path, 'dry_run'))
            continue

        try:
            returncode = run_experiment(cmd)
            if returncode == 0:
                print(f"{status_prefix} DONE  {run_name}")
                results.append((config_path, 'done'))
            else:
                print(f"{status_prefix} FAILED {run_name}  (exit code {returncode})")
                results.append((config_path, f'failed: exit {returncode}'))
        except KeyboardInterrupt:
            print(f"\n{status_prefix} INTERRUPTED by user.")
            results.append((config_path, 'interrupted'))
            break
        except Exception as e:
            print(f"{status_prefix} FAILED {run_name}: {e}")
            traceback.print_exc()
            results.append((config_path, f'failed: {e}'))

    # Summary
    print("\n── Queue Summary ─────────────────────────────────────────────")
    for path, status in results:
        print(f"  {status:<25} {path}")
    print()


if __name__ == '__main__':
    main()
