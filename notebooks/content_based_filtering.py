import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DATA_PATH = "data/processed"
OUTPUT_PATH = DATA_PATH  # we'll save npy files here
os.makedirs(OUTPUT_PATH, exist_ok=True)


products = pd.read_csv(os.path.join(DATA_PATH, "products_processed.csv"))

# For local safety — sample a subset for quick testing
n_samples = min(1000, len(products))
products_sampled = products.sample(n_samples, random_state=42).reset_index(drop=True)

# Combine text features into one column
def combine_features(row):
    return " ".join([
        str(row.get("title", "")),
        str(row.get("description", "")),
        str(row.get("brand", "")),
    ])

products_sampled["combined_text"] = products_sampled.apply(combine_features, axis=1)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(products_sampled["combined_text"])

# Compute cosine similarity (in manageable batches)
batch_size = 300
n_items = tfidf_matrix.shape[0]
top_k = 5

topk_indices = np.zeros((n_items, top_k), dtype=np.int32)
topk_similarities = np.zeros((n_items, top_k), dtype=np.float32)

for start in range(0, n_items, batch_size):
    end = min(start + batch_size, n_items)
    batch_sim = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
    np.fill_diagonal(batch_sim, 0)
    topk_idx = np.argsort(-batch_sim, axis=1)[:, :top_k]
    topk_val = np.take_along_axis(batch_sim, topk_idx, axis=1)
    topk_indices[start:end] = topk_idx
    topk_similarities[start:end] = topk_val

# Save results for Phase 6
np.save(os.path.join(OUTPUT_PATH, "content_topk_indices.npy"), topk_indices)
np.save(os.path.join(OUTPUT_PATH, "content_topk_similarities.npy"), topk_similarities)

# Example: Recommend similar products for a sample product
sample_idx = 0
sample_product = products_sampled.iloc[sample_idx]
print(f"\nRecommendations for product: {sample_product['asin']} - {sample_product['title'][:60]}...\n")

for idx, score in zip(topk_indices[sample_idx], topk_similarities[sample_idx]):
    rec_product = products_sampled.iloc[idx]
    print(f"{rec_product['asin']} - {rec_product['title'][:60]} (similarity: {score:.4f})")

print("\nPhase 5 Complete: Content-based model built successfully (sampled).")
