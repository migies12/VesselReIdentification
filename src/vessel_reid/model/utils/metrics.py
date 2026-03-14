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


def compute_map(
    similarity_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> float:
    """
    mean avg precision: for each query, do avg precision over
    the ranked gallery list, then avg across queries

    Args:
        similarity_matrix: shape (num_queries, num_gallery), higher = more similar
        query_ids: shape (num_queries,)
        gallery_ids: shape (num_gallery,)

    returns map score, which is 0-1
    """
    num_queries = similarity_matrix.shape[0]
    average_precisions = []

    for i in range(num_queries):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        sorted_gallery_ids = gallery_ids[sorted_indices]
        relevant = (sorted_gallery_ids == query_ids[i])
        num_relevant = relevant.sum()

        if num_relevant == 0:
            continue

        # precision at each rank where correct match is found
        cumulative_correct = np.cumsum(relevant)
        ranks = np.arange(1, len(sorted_gallery_ids) + 1)
        precision_at_k = cumulative_correct / ranks
        average_precision = (precision_at_k * relevant).sum() / num_relevant
        average_precisions.append(average_precision)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def compute_f1_at_threshold(
    similarity_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    threshold: float,
) -> dict:
    """
    compute a bunch of helpful metrics for comparing different models: 
    precision, recall, F1, FPR, and FNR at a given similarity threshold

    query-gallery pair above threshold counts as predicted match

    Args:
        similarity_matrix: shape (num_queries, num_gallery), higher = more similar
        query_ids: shape (num_queries,)
        gallery_ids: shape (num_gallery,)
        threshold: minimum similarity to accept a match

    gives dict with: precision, recall, f1, fpr, fnr, tp, fp, fn, tn
    """
    tp = fp = fn = tn = 0

    for i in range(len(query_ids)):
        for j in range(len(gallery_ids)):
            predicted_match = similarity_matrix[i, j] >= threshold
            actual_match = query_ids[i] == gallery_ids[j]

            if predicted_match and actual_match:
                tp += 1
            elif predicted_match and not actual_match:
                fp += 1
            elif not predicted_match and actual_match:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_optimal_threshold(
    similarity_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    max_fpr: float = 0.01,
    step: float = 0.01,
) -> dict:
    """
    tries a bunch of thresholds to find lowest theshold where fpr is at or below max 
    goes high to low 
    this will help us prioritize avoiding false positives, and only accept confident matches 

    if no threshold has fpr <= max fpr, returns threshold with lowest fpr

    args:
        similarity_matrix: shape (num_queries, num_gallery), higher = more similar
        query_ids: shape (num_queries,)
        gallery_ids: shape (num_gallery,)
        max_fpr: max acceptable false positive rate  (fpr) (default 0.01)
        step: threshold increment to sweep (default 0.01)

    returns dict w threshold, precision, recall, f1, fpr, fnr
    """
    thresholds = np.arange(1.0, 0.0 - step, -step)
    best = None

    best_valid = None   # best threshold satisfying max fpr constraint
    best_fallback = None  # lowest fpr seen if constraint never good

    for t in thresholds:
        t = round(float(t), 4)
        metrics = compute_f1_at_threshold(similarity_matrix, query_ids, gallery_ids, t)
        entry = {"threshold": t, **{k: metrics[k] for k in ("precision", "recall", "f1", "fpr", "fnr")}}
        if metrics["fpr"] <= max_fpr:
            # keep going lower to maximise recall while fpr stays acceptable
            best_valid = entry
        elif best_fallback is None or metrics["fpr"] < best_fallback["fpr"]:
            best_fallback = entry

    return best_valid if best_valid is not None else best_fallback
