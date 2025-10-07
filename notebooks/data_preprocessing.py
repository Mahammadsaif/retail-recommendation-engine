import pandas as pd
import numpy as np
import os
import json
from scipy.sparse import csr_matrix, save_npz

RAW_RATINGS_PATH = "data/raw/ratings_Electronics.csv"
RAW_PRODUCTS_PATH = "data/raw/Amazon_Electronics.csv"

def load_raw_data():
    ratings = pd.read_csv(RAW_RATINGS_PATH, names=['user_id', 'product_id', 'rating', 'timestamp'])
    products = pd.read_csv(RAW_PRODUCTS_PATH)
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    return ratings, products

def filter_cold_start(ratings):
    user_counts = ratings['user_id'].value_counts()
    product_counts = ratings['product_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= 5].index)]
    ratings = ratings[ratings['product_id'].isin(product_counts[product_counts >= 5].index)]
    return ratings

def clean_products(products):
    products = products.dropna(subset=['asin', 'title'])
    products['description'] = products['description'].fillna('')
    products['brand'] = products['brand'].fillna('Unknown')
    products['price'] = products['price'].fillna(0)
    return products

def encode_ids(ratings):
    user2id = {uid: idx for idx, uid in enumerate(ratings['user_id'].unique())}
    product2id = {pid: idx for idx, pid in enumerate(ratings['product_id'].unique())}
    ratings['user_idx'] = ratings['user_id'].map(user2id)
    ratings['product_idx'] = ratings['product_id'].map(product2id)
    return ratings, user2id, product2id

def build_sparse_matrix(ratings):
    matrix = csr_matrix((ratings['rating'], (ratings['user_idx'], ratings['product_idx'])))
    return matrix

def main():
    os.makedirs("data/processed", exist_ok=True)
    ratings, products = load_raw_data()
    ratings = filter_cold_start(ratings)
    products = clean_products(products)
    ratings, user2id, product2id = encode_ids(ratings)

    train = ratings.sample(frac=0.8, random_state=42)
    test = ratings.drop(train.index)

    train_matrix = build_sparse_matrix(train)
    test_matrix = build_sparse_matrix(test)

    save_npz("data/processed/train_matrix.npz", train_matrix)
    save_npz("data/processed/test_matrix.npz", test_matrix)

    ratings.to_csv("data/processed/ratings_processed.csv", index=False)
    products.to_csv("data/processed/products_processed.csv", index=False)

    with open("data/processed/user2id.json", "w") as f:
        json.dump(user2id, f)
    with open("data/processed/product2id.json", "w") as f:
        json.dump(product2id, f)

    print("Phase 2 Complete: Data preprocessing done.")

if __name__ == "__main__":
    main()
