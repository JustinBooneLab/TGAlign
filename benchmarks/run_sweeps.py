"""
Parameter Sweeps for TGAlign
============================
Generates publication-grade figures for:
1. Figure S1: Distance Threshold (Accuracy vs Rejection Rate)
2. Figure S2: K-mer Size Sweep
"""

import os
import sys
import gzip
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Ensure we import the installed package, not the local uncompiled folder
try:
    from tgalign import TGAlignIndex
except ImportError:
    print("Error: tgalign not installed. Run 'pip install .'")
    sys.exit(1)


# =============================================================================
# 1. DATA LOADING (Exact match to Main Benchmark COI Full-Length)
# =============================================================================

def parse_ncbi_taxonomy(header: str):
    parts = header.split()
    for i, part in enumerate(parts):
        if i + 1 < len(parts) and part[0].isupper() and len(part) > 2 and parts[i + 1][0].islower():
            genus = part
            species = f"{part} {parts[i + 1]}"
            if i + 2 < len(parts) and parts[i + 2][0].islower():
                species += f" {parts[i + 2]}"
            return species, genus
    return "Unknown", "Unknown"


def get_coi_full():
    url = "https://zenodo.org/records/17973054/files/ncbi_coi.fasta?download=1"
    os.makedirs("data_cache", exist_ok=True)
    path = "data_cache/ncbi_coi.fasta"

    if not os.path.exists(path):
        print("Downloading COI data...")
        import subprocess
        subprocess.run(f"curl -L -s '{url}' -o {path}", shell=True)

    # Read FASTA
    raw_dict = {}
    _open = gzip.open if path.endswith((".gz", ".gzip")) else open
    with _open(path, 'rt', errors='ignore') as f:
        header = None;
        curr = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if header: raw_dict[header] = "".join(curr)
                header = line[1:];
                curr = []
            else:
                curr.append(line.upper().replace('U', 'T'))
        if header: raw_dict[header] = "".join(curr)

    raw_s, raw_l = [], []
    for header, seq in raw_dict.items():
        if 'N' in seq: continue
        sp, _ = parse_ncbi_taxonomy(header)
        if sp != "Unknown":
            # COI Length bounds
            if 600 <= len(seq) <= 700:
                raw_s.append(seq)
                raw_l.append(sp)

    print(f"Loaded {len(raw_s)} COI Full-Length sequences.")
    return np.array(raw_s), np.array(raw_l)


# =============================================================================
# 2. SWEEP FUNCTIONS
# =============================================================================

def run_threshold_sweep(seqs, lbls, skf):
    print("\n--- Running Distance Threshold Sweep ---")
    thresholds = np.arange(0.5, 1.05, 0.02)

    # We only need to run Fold 1 for the threshold curve to get a smooth representative line
    train_idx, test_idx = next(skf.split(seqs, lbls))
    train_db = {f"{l.replace(' ', '_')}_{i}": s for i, (s, l) in enumerate(zip(seqs[train_idx], lbls[train_idx]))}
    queries = list(seqs[test_idx])
    ground_truth = [l.replace(' ', '_') for l in lbls[test_idx]]

    # Build model ONCE with a massive threshold, then we manually filter distances
    # to save extreme amounts of compute time.
    model = TGAlignIndex(k=11, s=9, distance_threshold=2.0)
    model.build(train_db)

    # Get raw distances and labels from FAISS
    query_sketches = model._sketch_batch(queries)
    distances, labels = model.index.search(query_sketches, k=1)

    acc_means = []
    rej_rates = []

    for t in thresholds:
        correct = 0
        assigned = 0

        for i in range(len(queries)):
            dist = distances[i][0]
            idx = labels[i][0]

            if idx != -1 and dist < t:
                assigned += 1
                pred_id = model.label_map[idx]
                pred_clean = pred_id.rsplit('_', 1)[0] if '_' in pred_id else pred_id
                if pred_clean == ground_truth[i]:
                    correct += 1

        acc = (correct / assigned) * 100 if assigned > 0 else 0.0
        rej = (1.0 - (assigned / len(queries))) * 100

        acc_means.append(acc)
        rej_rates.append(rej)
        print(f"Threshold: {t:.2f} | Accuracy: {acc:.2f}% | Rejection: {rej:.2f}%")

    # --- PLOT FIGURE S1 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Distance Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy of Assignments (%)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(thresholds, acc_means, color=color1, linewidth=2.5, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(85, 101)  # Zoom in on the top accuracy to show the drop

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Rejection Rate (%)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(thresholds, rej_rates, color=color2, linewidth=2.5, linestyle='--', label='Rejection Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)

    # Optimal point line
    ax1.axvline(x=0.8, color='black', linestyle=':', linewidth=2)
    ax1.text(0.81, 86, 'Selected Threshold (0.8)', fontsize=11, fontweight='bold')

    plt.title('Threshold Optimization for TGAlign on COI Dataset', fontsize=14, fontweight='bold')
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/Figure_S1_Threshold_Sweep.pdf', dpi=300)
    plt.savefig('figures/Figure_S1_Threshold_Sweep.png', dpi=300)
    print("Saved Figure S1 to figures/Figure_S1_Threshold_Sweep.pdf")


def run_kmer_sweep(seqs, lbls, skf):
    print("\n--- Running K-mer Size Sweep (5-Fold CV) ---")
    k_values = [9, 11, 13, 15]

    k_means = []
    k_stds = []

    import warnings
    warnings.filterwarnings("ignore")  # Ignore sklearn split warnings for rare classes

    for k_val in k_values:
        s_val = k_val - 2
        print(f"Evaluating k={k_val}, s={s_val}...")

        fold_accs = []
        for train_idx, test_idx in skf.split(seqs, lbls):
            train_db = {f"{l.replace(' ', '_')}_{i}": s for i, (s, l) in
                        enumerate(zip(seqs[train_idx], lbls[train_idx]))}
            queries = list(seqs[test_idx])
            ground_truth = [l.replace(' ', '_') for l in lbls[test_idx]]

            # Build and search
            model = TGAlignIndex(k=k_val, s=s_val, distance_threshold=0.8)
            model.build(train_db)
            preds = model.search(queries)

            fold_accs.append(accuracy_score(ground_truth, preds) * 100)

        k_means.append(np.mean(fold_accs))
        k_stds.append(np.std(fold_accs))
        print(f"  Result: {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%")

    # --- PLOT FIGURE S2 ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(k_values, k_means, yerr=k_stds, fmt='-o', color='tab:blue',
                 linewidth=2.5, markersize=8, capsize=5, capthick=2, label='Mean Accuracy (5-fold CV)')

    # Highlight selected point
    plt.axvline(x=11, color='tab:red', linestyle='--', linewidth=2, label='Selected k=11')

    plt.xlabel('k-mer size (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Classification Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Parameter Sweep for Syncmer k-mer size (s = k-2)', fontsize=14, fontweight='bold')
    plt.xticks(k_values)
    plt.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    plt.savefig('figures/Figure_S2_Kmer_Sweep.pdf', dpi=300)
    plt.savefig('figures/Figure_S2_Kmer_Sweep.png', dpi=300)
    print("Saved Figure S2 to figures/Figure_S2_Kmer_Sweep.pdf")


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # Load data once for both sweeps
    seqs, lbls = get_coi_full()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    run_threshold_sweep(seqs, lbls, skf)
    run_kmer_sweep(seqs, lbls, skf)
    print("\nAll sweeps complete. High-resolution figures generated in the 'figures/' directory.")