import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_PATH = "data/processed"

def load_data():
    train_matrix = sp.load_npz(os.path.join(PROCESSED_PATH, "train_matrix.npz"))
    test_matrix = sp.load_npz(os.path.join(PROCESSED_PATH, "test_matrix.npz"))
    return train_matrix, test_matrix

def compute_topk_items(train_matrix, k=50, batch_size=5000):
    n_items = train_matrix.shape[1]
    topk_indices = np.zeros((n_items, k), dtype=np.int32)
    topk_similarities = np.zeros((n_items, k), dtype=np.float32)

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = train_matrix[:, start:end].T
        sim = cosine_similarity(batch, train_matrix.T, dense_output=False)
        sim = sim.toarray()
        np.fill_diagonal(sim, 0)
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_val = np.take_along_axis(sim, topk_idx, axis=1)
        topk_indices[start:end] = topk_idx
        topk_similarities[start:end] = topk_val

    return topk_indices, topk_similarities

def main():
    train_matrix, test_matrix = load_data()
    topk_indices, topk_similarities = compute_topk_items(train_matrix)

    np.save(os.path.join(PROCESSED_PATH, "topk_indices.npy"), topk_indices)
    np.save(os.path.join(PROCESSED_PATH, "topk_similarities.npy"), topk_similarities)

    print("Phase 3 Complete: Collaborative filtering model built.")

if __name__ == "__main__":
    main()
