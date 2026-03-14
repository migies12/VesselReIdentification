import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def rank1_accuracy(distances: np.ndarray, labels: np.ndarray, query_labels: np.ndarray) -> float:
    top1 = np.argmin(distances, axis=1)
    matches = labels[top1] == query_labels
    return float(matches.mean())


def compute_rankk_accuracy(
    similarity_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    k: int,
) -> float:
    """
    Rank-k accuracy: fraction of queries where the correct ID shows up in top-k results.

    Args:
        similarity_matrix: shape (num_queries, num_gallery), higher = more similar
        query_ids: shape (num_queries,)
        gallery_ids: shape (num_gallery,)
        k: number of top results to consider
    """
    num_queries = similarity_matrix.shape[0]
    top_k_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
    correct = 0
    for i in range(num_queries):
        top_k_ids = gallery_ids[top_k_indices[i]]
        if query_ids[i] in top_k_ids:
            correct += 1
    return correct / num_queries
