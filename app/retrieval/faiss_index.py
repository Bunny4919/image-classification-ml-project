import os
import json
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np


class FAISSGalleryIndex:
    """FAISS vector index wrapper supporting IndexFlatIP (cosine similarity) and metadata tracking."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_item(self, vector: np.ndarray, meta: Dict[str, Any]):
        """Adds a single vector and corresponding metadata dict."""
        vec = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        self.index.add(vec)
        self.metadata.append(meta)

    def add_batch(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        """Adds a batch of vectors (N, dim) and corresponding metadata list."""
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vecs)
        self.metadata.extend(metas)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches top_k closest matches for a query vector.
        Returns list of dicts: {"score": float, "metadata": Dict}.
        """
        if self.index.ntotal == 0:
            return []

        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "score": float(dist),
                    "metadata": self.metadata[idx]
                })

        return results

    def save(self, index_dir: str):
        """Saves FAISS index binary and metadata JSON to directory."""
        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, "faiss_gallery.index")
        meta_file = os.path.join(index_dir, "faiss_gallery_meta.json")

        faiss.write_index(self.index, index_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "metadata": self.metadata
            }, f, indent=2)

        print(f"Saved FAISS index ({self.index.ntotal} vectors, {self.dimension}d) to {index_dir}")

    @classmethod
    def load(cls, index_dir: str) -> "FAISSGalleryIndex":
        """Loads FAISS index binary and metadata from directory."""
        index_file = os.path.join(index_dir, "faiss_gallery.index")
        meta_file = os.path.join(index_dir, "faiss_gallery_meta.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"FAISS index files not found in {index_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        dimension = meta_data.get("dimension", 384)
        gallery_index = cls(dimension=dimension)
        gallery_index.index = faiss.read_index(index_file)
        gallery_index.metadata = meta_data.get("metadata", [])

        print(f"Loaded FAISS index with {gallery_index.index.ntotal} vectors from {index_dir}")
        return gallery_index
