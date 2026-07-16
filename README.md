# TGAlign: Task-Geometry Alignment for DNA Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17973054.svg)](https://doi.org/10.5281/zenodo.17973054)

**TGAlign** is an alignment-free DNA sequence search engine that solves the geometric and computational bottlenecks of modern bioinformatics by projecting biological homology into high-dimensional vector space.

By replacing traditional sequence alignment with **Syncmer-based vectorization** and **Approximate Nearest Neighbor (ANN)** search, TGAlign autonomously resolves fragment-to-reference asymmetries and achieves **statistically superior accuracy** on indel-heavy sequences, while executing queries in fractions of a millisecond.

### Key Features
*   **🚀 C++ Accelerated:** Core sketching algorithms are implemented in optimized C++ with Pybind11 bindings, reducing sequence comparison to high-throughput matrix operations.
*   **🧬 Indel Robustness:** Utilizes **Strand-Symmetric Syncmers** (Edgar, 2021) instead of standard minimizers, preventing accuracy collapse on gap-heavy markers (e.g., 16S, ITS).
*   **🧩 Parameter-Free Geometry:** Automatically detects sequence length context to decompose references ("Task-Geometry Alignment"), solving the "Fragment Problem" without requiring user-tuned coverage heuristics.
*   **📉 Sub-Millisecond Latency:** Bypasses $O(N^2)$ dynamic programming, achieving a 2x to 4x latency reduction over highly optimized heuristic aligners.

---

## 📊 Performance Benchmark

TGAlign was benchmarked against leading state-of-the-art aligners (**USEARCH v12, VSEARCH, and MMseqs2**) across 5 challenging datasets using rigorous 5-fold stratified cross-validation. 

| Dataset | Biological Task | SOTA Baseline (Accuracy) | TGAlign (Ours) | Difference |
| :--- | :--- | :--- | :--- | :--- |
| **16S V4** | Indel-Heavy | 75.67% *(VSEARCH)* | **86.15%** | **+10.4%** |
| **COI Fragments** | Length Asymmetry | 71.50% *(MMseqs2)* | **72.67%** | **+1.1%** |
| **ITS (Fungi)** | Extreme Variation | 47.72% *(MMseqs2)* | **47.37%** | (Parity, p > 0.5) |
| **COI Full** | Point Mutations | 71.28% *(VSEARCH)* | **71.23%** | (Parity, p > 0.8) |

> **Validation:** The experimental design and benchmarking protocols were vetted by **Robert Edgar** (developer of USEARCH/MUSCLE).

---

## 📦 Installation

### Prerequisites
*   Python 3.8+
*   C++17 compliant compiler (GCC/Clang)
*   `faiss-cpu` (or `faiss-gpu`)

### From Source
```bash
git clone https://github.com/JustinBooneLab/TGAlign.git
cd TGAlign
pip install -v .