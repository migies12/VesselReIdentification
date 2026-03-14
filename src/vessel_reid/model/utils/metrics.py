import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def rank1_accuracy(distances: np.ndarray, labels: np.ndarray, query_labels: np.ndarray) -> float:
    top1 = np.argmin(distances, axis=1)
    matches = labels[top1] == query_labels
    return float(matches.mean())
