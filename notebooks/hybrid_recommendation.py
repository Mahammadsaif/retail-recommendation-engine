"""
Phase 6: Hybrid Recommendation System
Combines Collaborative Filtering and Content-Based Filtering
to produce balanced, smarter recommendations.
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "data/processed"
os.makedirs(DATA_PATH, exist_ok=True)

# --------------------------------------------------
# Load collaborative filtering data (Phase 3)
# --------------------------------------------------
collab_indices = np.load(os.path.join(DATA_PATH, "topk_indices.npy"))
collab_similarities = np.load(os.path.join(DATA_PATH, "topk_similarities.npy"))

# --------------------------------------------------
# Load content-based data (Phase 5)
# --------------------------------------------------
content_indices = np.load(os.path.join(DATA_PATH, "content_topk_indices.npy"))
content_similarities = np.load(os.path.join(DATA_PATH, "content_topk_similarities.npy"))

# --------------------------------------------------
# Load product info (for readability)
# --------------------------------------------------
products = pd.read_csv(os.path.join(DATA_PATH, "products_processed.csv"))

# --------------------------------------------------
# Hybrid Recommendation Function
# --------------------------------------------------
def hybrid_recommend(product_index, top_n=5, alpha=0.6):
    """
    Combine recommendations from both models:
    - alpha = weight for collaborative filtering
    - (1 - alpha) = weight for content-based filtering
    """
    # Collaborative filtering part
    collab_rec = collab_indices[product_index][:top_n]
    collab_scores = collab_similarities[product_index][:top_n]

    # Content-based part
    content_rec = content_indices[product_index][:top_n]
    content_scores = content_similarities[product_index][:top_n]

    # Combine indices and scores
    combined = {}
    for idx, score in zip(collab_rec, collab_scores):
        combined[idx] = combined.get(idx, 0) + alpha * score
    for idx, score in zip(content_rec, content_scores):
        combined[idx] = combined.get(idx, 0) + (1 - alpha) * score

    # Sort combined recommendations by score
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]

# --------------------------------------------------
# Example: Generate Hybrid Recommendations
# --------------------------------------------------
sample_index = 0
sample_product = products.iloc[sample_index]
print(f"\nHybrid Recommendations for product: {sample_product['asin']} - {sample_product['title'][:60]}...\n")

recommendations = hybrid_recommend(sample_index, top_n=5, alpha=0.6)

for idx, score in recommendations:
    rec_product = products.iloc[idx]
    print(f"{rec_product['asin']} - {rec_product['title'][:60]} (hybrid score: {score:.4f})")

print("\nHybrid Recommendation System built successfully.")
