import os
import numpy as np
import pandas as pd

PROCESSED_PATH = "data/processed"

# Load processed ratings and products
ratings = pd.read_csv(os.path.join(PROCESSED_PATH, "ratings_processed.csv"))
products = pd.read_csv(os.path.join(PROCESSED_PATH, "products_processed.csv"))

# Load collaborative filtering matrices from Phase 3
topk_indices = np.load(os.path.join(PROCESSED_PATH, "topk_indices.npy"))
topk_similarities = np.load(os.path.join(PROCESSED_PATH, "topk_similarities.npy"))

# Build product ID ↔ index mapping
unique_products = ratings["product_id"].unique()
product_to_index = {pid: idx for idx, pid in enumerate(unique_products)}
index_to_product = {idx: pid for pid, idx in product_to_index.items()}

# Function to recommend items
def recommend_items(product_id, top_n=5):
    if product_id not in product_to_index:
        print("Product not found in training data.")
        return []

    idx = product_to_index[product_id]
    similar_indices = topk_indices[idx][:top_n]
    similar_scores = topk_similarities[idx][:top_n]

    recommendations = []
    for sim_idx, score in zip(similar_indices, similar_scores):
        pid = index_to_product.get(sim_idx, None)
        if pid:
            recommendations.append((pid, score))
    return recommendations

# Example
sample_product = ratings["product_id"].iloc[0]
print(f"\nRecommendations for product: {sample_product}\n")

recs = recommend_items(sample_product, top_n=5)
for pid, score in recs:
    product_name = products.loc[products["asin"] == pid, "title"].values
    product_name = product_name[0] if len(product_name) > 0 else "Unknown"
    print(f"{pid} - {product_name} (similarity: {score:.4f})")

print("\nPhase 4 Complete: Recommendations generated successfully.")
