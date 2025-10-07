"""
Phase 2: Data Preprocessing
===========================

Goal:
- Filter cold start users and products
- Handle missing values in product metadata
- Prepare datasets for training collaborative and hybrid models
- Save processed datasets
"""

import pandas as pd
import numpy as np
import os
import json

# ==================== 1. LOAD RAW DATA ====================
print("="*60)
print("🚀 PHASE 2: DATA PREPROCESSING")
print("="*60)

ratings_df = pd.read_csv('data/raw/ratings_Electronics.csv', 
                         names=['user_id', 'product_id', 'rating', 'timestamp'])
ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

products_df = pd.read_csv('data/raw/Amazon_Electronics.csv')

print(f"Ratings shape: {ratings_df.shape}")
print(f"Products shape: {products_df.shape}")

# ==================== 2. FILTER COLD START ====================
# Users with <5 ratings, products with <5 ratings are considered cold start
#Counting, number of ratings per user and per product
user_counts = ratings_df['user_id'].value_counts()
product_counts = ratings_df['product_id'].value_counts()

ratings_df = ratings_df[ratings_df['user_id'].isin(user_counts[user_counts >= 5].index)]
ratings_df = ratings_df[ratings_df['product_id'].isin(product_counts[product_counts >= 5].index)]

print(f"After filtering cold start users/products:")
print(f"Ratings shape: {ratings_df.shape}")

# ==================== 3. HANDLE MISSING VALUES ====================
# For simplicity, drop products with missing essential info
products_df = products_df.dropna(subset=['asin', 'title'])
products_df['description'] = products_df['description'].fillna('')
products_df['brand'] = products_df['brand'].fillna('Unknown')
products_df['price'] = products_df['price'].fillna(0)

print("Missing values handled in product metadata.")

# ==================== 4. ENCODE USERS AND PRODUCTS ====================
# Map IDs to integer indices for matrix factorization
user2id = {uid: idx for idx, uid in enumerate(ratings_df['user_id'].unique())}
product2id = {pid: idx for idx, pid in enumerate(ratings_df['product_id'].unique())}

ratings_df['user_idx'] = ratings_df['user_id'].map(user2id)
ratings_df['product_idx'] = ratings_df['product_id'].map(product2id)

print("User and product IDs encoded for ML models.")

# ==================== 5. SAVE PROCESSED DATA ====================
os.makedirs('data/processed', exist_ok=True)

ratings_df.to_csv('data/processed/ratings_processed.csv', index=False)
products_df.to_csv('data/processed/products_processed.csv', index=False)

# Save encoders
with open('data/processed/user2id.json', 'w') as f:
    json.dump(user2id, f)

with open('data/processed/product2id.json', 'w') as f:
    json.dump(product2id, f)

print("Processed datasets and encoders saved in 'data/processed/'")
print("="*60)
print("🎯 PHASE 2 COMPLETE: Data Preprocessing Done")
print("="*60)
