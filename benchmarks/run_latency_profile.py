import os
import sys

import time
import numpy as np

try:
    from tgalign import TGAlignIndex
except ImportError:
    print("Error: tgalign not installed. Run 'pip install -e .'")
    sys.exit(1)

def run_profiler():
    # 1. Generate dummy data that mimics COI (~650bp)
    np.random.seed(42)
    bases = ['A', 'C', 'G', 'T']
    def make_seq(length): return "".join(np.random.choice(bases, length))

    print("Generating synthetic COI-like data...")
    ref_db = {f"ref_{i}": make_seq(650) for i in range(5000)}
    queries = [make_seq(650) for _ in range(1000)]

    # 2. Build the Model (Using the exact final architecture)
    print("Building index (IVFFlat, nprobe=5)...")
    model = TGAlignIndex(k=11, s=9, dim=4096, distance_threshold=0.8)
    model.build(ref_db)

    # 3. The Profiling Test
    print("\nRunning Latency Profiler on 1000 queries...")

    # A. Measure Feature Extraction (Sketching) Time
    t0 = time.perf_counter()
    query_sketches = model._sketch_batch(queries)
    t_sketch = time.perf_counter() - t0

    # B. Measure Vector Search (FAISS) Time
    t0 = time.perf_counter()
    distances, labels = model.index.search(query_sketches, k=1)
    t_search = time.perf_counter() - t0

    # C. Calculate Averages
    avg_sketch_ms = (t_sketch * 1000) / len(queries)
    avg_search_ms = (t_search * 1000) / len(queries)
    total_ms = avg_sketch_ms + avg_search_ms

    print("\n--- NEW LATENCY BREAKDOWN ---")
    print(f"Feature Extraction (C++ Syncmer Hashing): {avg_sketch_ms:.4f} ms ({avg_sketch_ms/total_ms*100:.1f}%)")
    print(f"Vector Search (FAISS IVFFlat, nprobe=5):  {avg_search_ms:.4f} ms ({avg_search_ms/total_ms*100:.1f}%)")
    print(f"Total Query Latency:                      {total_ms:.4f} ms")

if __name__ == "__main__":
    # Ensure you are running this in the activated 'env'
    run_profiler()