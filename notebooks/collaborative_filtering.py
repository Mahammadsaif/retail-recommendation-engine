"""
Phase 3 — Collaborative Filtering (clean)
- Load processed ratings + mappings
- Create temporal train/test split
- Build sparse train/test matrices (CSR)
- Compute top-K item neighbors (cosine) and store as sparse CSR
- Train TruncatedSVD and save factor matrices
- Provide simple recommenders (item-item top-K and SVD)

Notes:
- This script writes artifacts to data/processed/cf_artifacts
- Do NOT commit raw data to git. Keep 'data/' in .gitignore.
"""
from pathlib import Path
import logging
import json
import time
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, vstack, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# -- configuration --
DATA_DIR = Path("data/processed")
ARTIFACT_DIR = DATA_DIR / "cf_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed(ratings_path: Path, user_map_path: Path, product_map_path: Path) -> Tuple[pd.DataFrame, Dict[str,int], Dict[str,int]]:
    """Load ratings CSV and mapping JSONs saved in Phase 2."""
    if not ratings_path.exists():
        raise FileNotFoundError(f"{ratings_path} not found. Run Phase 2 preprocessing.")
    ratings = pd.read_csv(ratings_path)
    with open(user_map_path, "r") as f:
        user2id = json.load(f)
    with open(product_map_path, "r") as f:
        product2id = json.load(f)
    logger.info("Loaded processed data: ratings=%d, users=%d, products=%d", len(ratings), len(user2id), len(product2id))
    return ratings, user2id, product2id


def temporal_split(df: pd.DataFrame, timestamp_col: str = "timestamp", test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Global chronological split: last test_ratio fraction is test set."""
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    split = int(len(df_sorted) * (1 - test_ratio))
    train = df_sorted.iloc[:split].copy()
    test = df_sorted.iloc[split:].copy()
    logger.info("Temporal split: train=%d, test=%d", len(train), len(test))
    return train, test


def build_csr(df: pd.DataFrame, user2id: Dict[str,int], product2id: Dict[str,int]) -> csr_matrix:
    """
    Build CSR user×item matrix.
    Accepts dataframe with either (user_idx/product_idx) or (user_id/product_id).
    """
    if "user_idx" in df.columns and "product_idx" in df.columns:
        rows = df["user_idx"].astype(int).values
        cols = df["product_idx"].astype(int).values
    else:
        rows = df["user_id"].map(user2id).astype(int).values
        cols = df["product_id"].map(product2id).astype(int).values
    data = df["rating"].astype(float).values
    n_users = len(user2id)
    n_items = len(product2id)
    mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    logger.info("Built CSR matrix shape=%s nnz=%d", mat.shape, mat.nnz)
    return mat


# Instead of full pairwise similarity, compute top-k in batches
def compute_topk_similar_items(train_matrix, top_k=50, batch_size=500):
    """
    Computes top-k cosine similarity for items in manageable batches.
    """
    n_items = train_matrix.shape[1]
    topk_similarities = []
    topk_indices = []

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = train_matrix[:, start:end]
        sims = cosine_similarity(batch.T, train_matrix.T)
        
        # Get top-k for each item in batch
        topk_idx = np.argpartition(-sims, range(top_k), axis=1)[:, :top_k]
        topk_val = np.take_along_axis(sims, topk_idx, axis=1)

        topk_indices.append(topk_idx)
        topk_similarities.append(topk_val)

        logging.info(f"Processed items {start}–{end} of {n_items}")

    return np.vstack(topk_indices), np.vstack(topk_similarities)

logging.info("Computing top-50 item neighbors (cosine) in batches...")
topk_indices, topk_similarities = compute_topk_similar_items(train_matrix)
logging.info("Completed similarity computation.")



def recommend_item_item(user_id: str, user2id: Dict[str,int], product2id: Dict[str,int],
                        train_csr: csr_matrix, item_sim: csr_matrix, top_k: int = 10) -> List[tuple]:
    """Compute item-item recommendations for a user using precomputed item_sim CSR."""
    if user_id not in user2id:
        logger.warning("User %s not in user2id; returning empty.", user_id)
        return []
    uidx = user2id[user_id]
    user_row = train_csr.getrow(uidx)  # 1 x n_items
    if user_row.nnz == 0:
        return []
    # candidate_scores = user_row * item_sim  (1 x n_items)
    candidate = user_row.dot(item_sim).toarray().ravel()
    # mask already rated
    candidate[user_row.indices] = -np.inf
    top_idx = np.argpartition(-candidate, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-candidate[top_idx])]
    inv_product = {v: k for k, v in product2id.items()}
    return [(inv_product[int(i)], float(candidate[int(i)])) for i in top_idx if candidate[int(i)] != -np.inf]


def train_truncated_svd(train_csr: csr_matrix, n_components: int = 100, random_state: int = 42) -> Tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    """Train TruncatedSVD and return (model, user_factors, item_factors)."""
    logger.info("Training TruncatedSVD n_components=%d", n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    t0 = time.time()
    user_factors = svd.fit_transform(train_csr)     # n_users x k
    item_factors = svd.components_.T                # n_items x k
    t1 = time.time()
    logger.info("TruncatedSVD done in %.1fs", t1 - t0)
    return svd, user_factors, item_factors


def recommend_svd(user_id: str, user2id: Dict[str,int], product2id: Dict[str,int],
                  user_factors: np.ndarray, item_factors: np.ndarray, train_csr: csr_matrix, top_k: int = 10) -> List[tuple]:
    """Recommend top_k using dot product of user and item latent factors."""
    if user_id not in user2id:
        logger.warning("User %s not in user2id; returning empty.", user_id)
        return []
    uidx = user2id[user_id]
    uvec = user_factors[uidx]                     # k
    scores = item_factors.dot(uvec)               # n_items
    rated = train_csr.getrow(uidx).indices
    scores[rated] = -np.inf
    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    inv_product = {v: k for k, v in product2id.items()}
    return [(inv_product[int(i)], float(scores[int(i)])) for i in top_idx if scores[int(i)] != -np.inf]


def main():
    ratings_path = DATA_DIR / "ratings_processed.csv"
    user_map_path = DATA_DIR / "user2id.json"
    product_map_path = DATA_DIR / "product2id.json"

    ratings, user2id, product2id = load_processed(ratings_path, user_map_path, product_map_path)
    train_df, test_df = temporal_split(ratings, timestamp_col="timestamp", test_ratio=0.2)

    train_csr = build_csr(train_df, user2id, product2id)
    test_csr = build_csr(test_df, user2id, product2id)

    save_npz(ARTIFACT_DIR / "train_matrix.npz", train_csr)
    save_npz(ARTIFACT_DIR / "test_matrix.npz", test_csr)
    logger.info("Saved train/test matrices.")

    # Compute top-K item neighbors (memory-friendly)
    item_sim = compute_topk_item_neighbors(train_csr, top_k=50, n_jobs=4)
    joblib.dump(item_sim, ARTIFACT_DIR / "item_sim_topk.joblib")
    logger.info("Saved item_sim_topk.joblib")

    # Train SVD
    svd_model, user_factors, item_factors = train_truncated_svd(train_csr, n_components=100)
    joblib.dump(svd_model, ARTIFACT_DIR / "svd_model.joblib")
    joblib.dump(user_factors, ARTIFACT_DIR / "user_factors.npy")
    joblib.dump(item_factors, ARTIFACT_DIR / "item_factors.npy")
    logger.info("Saved SVD model and factors.")

    # smoke test
    sample_user = next(iter(user2id.keys()))
    recs_item = recommend_item_item(sample_user, user2id, product2id, train_csr, item_sim, top_k=5)
    recs_svd = recommend_svd(sample_user, user2id, product2id, user_factors, item_factors, train_csr, top_k=5)
    logger.info("Sample user=%s item-item recs=%s", sample_user, recs_item[:3])
    logger.info("Sample user=%s svd recs=%s", sample_user, recs_svd[:3])
    logger.info("Phase 3 pipeline finished; artifacts in %s", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
