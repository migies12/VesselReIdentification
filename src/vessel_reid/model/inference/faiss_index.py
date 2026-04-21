import json
import os
from typing import List, Tuple

import faiss
import numpy as np


def build_index(embeddings: np.ndarray, normalize: bool = True) -> faiss.IndexFlatIP:
    if normalize:
        faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, index_path: str) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


def load_index(index_path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(index_path)


def save_metadata(metadata: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search(index: faiss.IndexFlatIP, query: np.ndarray, top_k: int = 5, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if normalize:
        faiss.normalize_L2(query)
    distances, indices = index.search(query, top_k)
    return distances, indices
