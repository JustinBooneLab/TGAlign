import numpy as np
import faiss
from typing import List, Dict, Tuple
from . import _tgalign_cpp # Imports the compiled C++ module

class TGAlignIndex:
    def __init__(self, k: int = 11, s: int = 9, dim: int = 4096, distance_threshold: float = 0.8):
        self.k = k
        self.s = s
        self.vector_dim = dim
        self.threshold = distance_threshold
        self.index = None
        self.label_map = {}
        
        # TGA Tiling Parameters
        self.window_size = 350
        self.window_step = 50

    def _sketch_batch(self, sequences: List[str]) -> np.ndarray:
        """Internal call to C++ backend."""
        return _tgalign_cpp.create_syncmer_sketches_batch(
            sequences, self.k, self.s, self.vector_dim
        )

    def build(self, reference_db: Dict[str, str], tile_sequences: bool = False):
        """
        Builds the FAISS index from a dictionary of {seq_id: sequence}.
        
        Args:
            reference_db: Dict mapping ID to DNA sequence.
            tile_sequences: If True, applies TGA windowing (for fragment tasks).
        """
        nlist = max(1, min(len(reference_db) * 5 // 40, 200))
        quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)

        sequences_to_sketch = []
        current_idx = 0

        for ref_id, seq in reference_db.items():
            if tile_sequences and len(seq) > self.window_size:
                # TGA: Windowing strategy for fragments
                for i in range(0, len(seq) - self.window_size + 1, self.window_step):
                    sequences_to_sketch.append(seq[i : i + self.window_size])
                    self.label_map[current_idx] = ref_id
                    current_idx += 1
            else:
                sequences_to_sketch.append(seq)
                self.label_map[current_idx] = ref_id
                current_idx += 1

        if not sequences_to_sketch:
            raise ValueError("No valid sequences found to index.")

        sketches = self._sketch_batch(sequences_to_sketch)
        
        # Train and Add to FAISS
        self.index.train(sketches)
        self.index.add(sketches)

    def search(self, query_sequences: List[str]) -> List[str]:
        """
        Searches the index for query sequences. Returns list of predicted IDs.
        """
        if not self.index or self.index.ntotal == 0:
            return ["Unknown"] * len(query_sequences)

        query_sketches = self._sketch_batch(query_sequences)
        distances, labels = self.index.search(query_sketches, k=1)

        results = []
        for i in range(len(query_sequences)):
            idx = labels[i][0]
            dist = distances[i][0]
            if idx != -1 and dist < self.threshold:
                full_id = self.label_map[idx]
                # Clean ID if it contains tiling suffixes or extra info
                results.append(full_id.rsplit('_', 1)[0] if '_' in full_id else full_id)
            else:
                results.append("Unknown")
        return results
