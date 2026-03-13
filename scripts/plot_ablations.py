"""
Plot ablation study results from TensorBoard logs.

Primary view: scatter plot of maximum sequence_accuracy (actual_acc) per run,
grouped by ablation group.  Optional line plots show the full training curve.

Usage:
    python scripts/plot_ablations.py                        # scatter of max acc
    python scripts/plot_ablations.py --mode line            # full training curves
    python scripts/plot_ablations.py --mode both            # scatter + line
    python scripts/plot_ablations.py --groups A T           # specific groups only
    python scripts/plot_ablations.py --metric test/accuracy # pixel-level instead
    python scripts/plot_ablations.py --smooth 5             # smoothing for line mode
    python scripts/plot_ablations.py --csv                  # also export CSV

Output:
    outputs/plots/
        scatter_overview.png     — max acc per run, all groups
        scatter_group_A.png      — max acc within Group A
        scatter_group_T.png      — max acc within Group T
        ...
        line_overview.png        — training curves, all groups  (--mode line/both)
        line_group_A.png         — training curves, Group A     (--mode line/both)
        results_<metric>.csv     — step-level data              (--csv)
"""
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── TensorBoard reader ────────────────────────────────────────────────────────

def read_tb_scalar(event_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """Return (steps, values) for one tag in a TensorBoard event directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError(
            "tensorboard is required. Install with: uv add tensorboard"
        )
    ea = EventAccumulator(str(event_dir), size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


# ── Group detection ───────────────────────────────────────────────────────────

_GROUP_RE = re.compile(r'^([A-Z]+)\d+')

def detect_group(run_name: str) -> str:
    m = _GROUP_RE.match(run_name)
    return m.group(1) if m else 'other'


GROUP_LABELS = {
    'A': 'Architecture\n(GPT vs TRM)',
    'B': 'Depth\n(n_layer)',
    'C': 'Embed dim\n(n_embd)',
    'D': 'Activation',
    'E': 'Normalization',
    'F': 'Pos encoding',
    'T': 'TRM cycles\n(H × L)',
}

METRIC_LABELS = {
    'test/sequence_accuracy': 'Puzzle Accuracy — actual_acc (exact match)',
    'test/accuracy':          'Pixel Accuracy',
    'test/loss':              'Test Loss',
    'train/loss':             'Train Loss',
}


def _apply_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.dpi': 130,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def _tab_colors(n: int) -> list:
    cmap = cm.get_cmap('tab10', max(n, 1))
    return [cmap(i) for i in range(n)]


def _group_color_map(groups: list[str]) -> dict[str, tuple]:
    return {g: c for g, c in zip(sorted(groups), _tab_colors(len(groups)))}


# ── Scatter plot (max value per run) ─────────────────────────────────────────

def _compute_max(runs: dict[str, tuple]) -> dict[str, tuple[float, int]]:
    """Return {run_name: (max_value, step_at_max)} for each run."""
    out = {}
    for run_name, (steps, values) in runs.items():
        if not values:
            continue
        idx = int(np.argmax(values))
        out[run_name] = (values[idx], steps[idx])
    return out


def plot_scatter_overview(
    all_runs: dict[str, tuple],
    metric: str,
    outpath: Path,
):
    """
    One dot per run.  X = position within a grouped layout (groups A, B, C …).
    Y = max metric value.  Groups separated by vertical dividers.
    """
    groups_sorted = sorted({detect_group(r) for r in all_runs})
    color_map = _group_color_map(groups_sorted)

    # Build ordered list: group by group, alphabetical within
    ordered: list[tuple[str, str]] = []  # [(group, run_name), ...]
    by_group: dict[str, list[str]] = defaultdict(list)
    for rn in all_runs:
        by_group[detect_group(rn)].append(rn)
    for g in groups_sorted:
        for rn in sorted(by_group[g]):
            ordered.append((g, rn))

    maxima = _compute_max(all_runs)

    xs = list(range(len(ordered)))
    ys = [maxima.get(rn, (0,))[0] for _, rn in ordered]
    colors = [color_map[g] for g, _ in ordered]
    labels = [rn for _, rn in ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.9 + 2), 5))

    ax.scatter(xs, ys, c=colors, s=120, zorder=3, edgecolors='white', linewidths=0.6)

    # Annotate each point with its run name and max value
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(f'{label}\n{y:.3f}',
                    xy=(x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, rotation=45)

    # Group dividers and group labels on x-axis
    group_starts: dict[str, int] = {}
    group_ends: dict[str, int] = {}
    for i, (g, _) in enumerate(ordered):
        if g not in group_starts:
            group_starts[g] = i
        group_ends[g] = i

    prev_end = -1
    for g in groups_sorted:
        start = group_starts[g]
        end = group_ends[g]
        mid = (start + end) / 2
        if start > 0:
            ax.axvline(start - 0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.text(mid, ax.get_ylim()[0] - 0.02, GROUP_LABELS.get(g, g),
                ha='center', va='top', fontsize=8,
                color=color_map[g], fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels([])          # labels drawn manually above
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f'Max {METRIC_LABELS.get(metric, metric)} — all ablations')
    ax.set_xlim(-0.8, len(ordered) - 0.2)
    ax.xaxis.grid(False)

    # Legend: one entry per group
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[g],
               markersize=9, label=GROUP_LABELS.get(g, g).replace('\n', ' '))
        for g in groups_sorted
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_scatter_group(
    runs: dict[str, tuple],
    group: str,
    metric: str,
    outpath: Path,
):
    """
    Scatter of max values for one ablation group.
    X = run name,  Y = max metric value.
    Also shows min/max range bar if multiple checkpoints exist.
    """
    maxima = _compute_max(runs)
    sorted_runs = sorted(maxima)
    colors = _tab_colors(len(sorted_runs))

    max_vals = [maxima[rn][0] for rn in sorted_runs]
    best_steps = [maxima[rn][1] for rn in sorted_runs]
    xs = list(range(len(sorted_runs)))

    fig, ax = plt.subplots(figsize=(max(5, len(sorted_runs) * 1.4 + 1.5), 4.5))

    # range bar: show full [min, max] span of the curve as a thin line
    for i, (rn, color) in enumerate(zip(sorted_runs, colors)):
        _, values = runs[rn]
        if len(values) > 1:
            ax.vlines(i, min(values), max(values),
                      color=color, linewidth=2, alpha=0.25, zorder=2)

    # scatter dot at max
    ax.scatter(xs, max_vals, c=colors, s=150, zorder=4,
               edgecolors='white', linewidths=0.8)

    # value label above dot
    for x, y, step in zip(xs, max_vals, best_steps):
        ax.annotate(f'{y:.3f}\n(iter {step:,})',
                    xy=(x, y), xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(sorted_runs, rotation=20, ha='right', fontsize=9)
    ax.xaxis.grid(False)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f'Group {group}: {GROUP_LABELS.get(group, group).replace(chr(10), " ")} — max {metric.split("/")[-1]}')

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ── Line plots (training curves) ──────────────────────────────────────────────

def smooth(values: list[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def plot_line_group(
    runs: dict[str, tuple],
    group: str,
    metric: str,
    smooth_window: int,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = _tab_colors(len(runs))

    for (run_name, (steps, values)), color in zip(sorted(runs.items()), colors):
        if not steps:
            continue
        y = smooth(values, smooth_window)
        ax.plot(steps, y, color=color, linewidth=1.8, label=run_name)
        if smooth_window > 1:
            ax.plot(steps, values, color=color, linewidth=0.4, alpha=0.2)
        # mark the max point
        idx = int(np.argmax(values))
        ax.scatter([steps[idx]], [values[idx]], color=color, s=60, zorder=5,
                   marker='*', edgecolors='white', linewidths=0.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f'Group {group}: {GROUP_LABELS.get(group, group).replace(chr(10), " ")}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_line_overview(
    all_runs: dict[str, tuple],
    metric: str,
    smooth_window: int,
    outpath: Path,
):
    groups = sorted({detect_group(r) for r in all_runs})
    color_map = _group_color_map(groups)
    seen: set[str] = set()

    fig, ax = plt.subplots(figsize=(12, 6))
    for run_name, (steps, values) in sorted(all_runs.items()):
        if not steps:
            continue
        g = detect_group(run_name)
        color = color_map[g]
        y = smooth(values, smooth_window)
        label = GROUP_LABELS.get(g, g).replace('\n', ' ') if g not in seen else '_nolegend_'
        seen.add(g)
        ax.plot(steps, y, color=color, linewidth=1.4, label=label, alpha=0.85)
        ax.annotate(run_name, xy=(steps[-1], y[-1]),
                    fontsize=6, color=color, alpha=0.7,
                    xytext=(3, 0), textcoords='offset points')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f'All ablations — {METRIC_LABELS.get(metric, metric)}')
    ax.legend(loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(all_runs: dict[str, tuple], metric: str, outpath: Path):
    import csv
    all_steps = sorted({s for steps, _ in all_runs.values() for s in steps})
    run_names = sorted(all_runs)
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step'] + run_names)
        for step in all_steps:
            row = [step]
            for rn in run_names:
                lookup = dict(zip(*all_runs[rn]))
                row.append(lookup.get(step, ''))
            writer.writerow(row)
    print(f"  Saved: {outpath}")


def export_max_csv(all_runs: dict[str, tuple], metric: str, outpath: Path):
    import csv
    maxima = _compute_max(all_runs)
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'group', 'max_value', 'step_at_max'])
        for rn in sorted(maxima):
            val, step = maxima[rn]
            writer.writerow([rn, detect_group(rn), f'{val:.6f}', step])
    print(f"  Saved: {outpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Plot ablation study results')
    parser.add_argument('--logdir', default='logs',
                        help='Root directory containing per-run TensorBoard logs')
    parser.add_argument('--outdir', default='outputs/plots',
                        help='Output directory for plots')
    parser.add_argument('--metric', default='test/sequence_accuracy',
                        help='TensorBoard scalar tag to plot')
    parser.add_argument('--mode', default='scatter',
                        choices=['scatter', 'line', 'both'],
                        help='scatter=max-value dots (default), line=training curves, both=both')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Restrict to specific group prefixes, e.g. A T')
    parser.add_argument('--smooth', type=int, default=1,
                        help='Rolling-average window for line mode (1 = off)')
    parser.add_argument('--csv', action='store_true',
                        help='Also export raw step data and max-value summary to CSV')
    return parser.parse_args()


def main():
    args = get_args()
    _apply_style()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not logdir.exists():
        print(f"ERROR: log directory not found: {logdir}")
        sys.exit(1)

    run_dirs = sorted([
        d for d in logdir.iterdir()
        if d.is_dir() and any(d.glob('events.out.*'))
    ])

    if not run_dirs:
        print(f"No TensorBoard event files found under {logdir}/")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s) under {logdir}/")
    print(f"Metric : {args.metric}")
    print(f"Mode   : {args.mode}")
    print()

    all_runs: dict[str, tuple] = {}
    for run_dir in run_dirs:
        run_name = run_dir.name
        if args.groups and detect_group(run_name) not in args.groups:
            continue
        steps, values = read_tb_scalar(run_dir, args.metric)
        if not steps:
            print(f"  WARN: '{args.metric}' not found in {run_name} — skipping")
            continue
        max_val = max(values)
        print(f"  Loaded {run_name:40s}  {len(steps)} pts  max={max_val:.4f}")
        all_runs[run_name] = (steps, values)

    if not all_runs:
        print("\nNo data loaded.")
        sys.exit(1)

    print()

    by_group: dict[str, dict] = defaultdict(dict)
    for rn, data in all_runs.items():
        by_group[detect_group(rn)][rn] = data

    do_scatter = args.mode in ('scatter', 'both')
    do_line    = args.mode in ('line', 'both')

    if do_scatter:
        plot_scatter_overview(all_runs, args.metric, outdir / 'scatter_overview.png')
        for group, runs in sorted(by_group.items()):
            plot_scatter_group(runs, group, args.metric,
                               outdir / f'scatter_group_{group}.png')

    if do_line:
        plot_line_overview(all_runs, args.metric, args.smooth,
                           outdir / 'line_overview.png')
        for group, runs in sorted(by_group.items()):
            plot_line_group(runs, group, args.metric, args.smooth,
                            outdir / f'line_group_{group}.png')

    if args.csv:
        slug = args.metric.replace('/', '_')
        export_max_csv(all_runs, args.metric, outdir / f'max_{slug}.csv')
        export_csv(all_runs, args.metric, outdir / f'all_{slug}.csv')

    print(f"\nDone. Plots written to {outdir}/")


if __name__ == '__main__':
    main()
