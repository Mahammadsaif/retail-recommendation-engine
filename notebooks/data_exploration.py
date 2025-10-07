"""
Phase 1: Data Exploration & Understanding
==========================================

This script explores the Amazon Electronics ratings dataset and product metadata.

Key Concepts:
- User-Item Matrix & Sparsity
- Long Tail Distribution
- Cold Start Problem
- Data Quality Assessment

Run this first to understand your data before building models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import json
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*60)
print("📊 PHASE 1: DATA EXPLORATION")
print("="*60)

# ==================== 1. LOAD RATINGS DATA ====================
print("\n1️⃣ Loading User-Product Interactions...")
print("-"*60)

# Load ratings data
# NOTE: Update column names if your CSV has headers
try:
    ratings_df = pd.read_csv('data/raw/ratings_Electronics.csv', 
                             names=['user_id', 'product_id', 'rating', 'timestamp'])
    print("✅ Ratings data loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: ratings_Electronics.csv not found in data/raw/")
    print("Please ensure the file is in the correct location.")
    exit()

print(f"\nShape: {ratings_df.shape[0]:,} interactions")
print(f"Columns: {list(ratings_df.columns)}")
print("\nFirst 5 rows:")
print(ratings_df.head())

# ==================== 2. BASIC STATISTICS ====================
print("\n\n2️⃣ Dataset Statistics")
print("-"*60)

n_users = ratings_df['user_id'].nunique()
n_products = ratings_df['product_id'].nunique()
n_interactions = len(ratings_df)

print(f"Total Interactions: {n_interactions:,}")
print(f"Unique Users: {n_users:,}")
print(f"Unique Products: {n_products:,}")
print(f"Rating Range: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
print(f"Average Rating: {ratings_df['rating'].mean():.2f}")
print(f"Median Rating: {ratings_df['rating'].median():.1f}")

# Convert timestamp to datetime
ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
print(f"\nDate Range: {ratings_df['datetime'].min()} to {ratings_df['datetime'].max()}")

# ==================== 3. DATA QUALITY CHECK ====================
print("\n\n3️⃣ Data Quality Check")
print("-"*60)

print("Missing Values:")
print(ratings_df.isnull().sum())
print(f"\nDuplicate Rows: {ratings_df.duplicated().sum()}")

# ==================== 4. SPARSITY ANALYSIS ====================
print("\n\n4️⃣ Sparsity Analysis (CRITICAL METRIC)")
print("-"*60)

possible_interactions = n_users * n_products
sparsity = 1 - (n_interactions / possible_interactions)

print(f"Matrix Size: {n_users:,} users × {n_products:,} products")
print(f"Possible Interactions: {possible_interactions:,}")
print(f"Actual Interactions: {n_interactions:,}")
print(f"\n🚨 SPARSITY: {sparsity*100:.4f}%")
print(f"   (Matrix is {sparsity*100:.2f}% empty!)")
print("\n💡 Why this matters:")
print("   - High sparsity = most users rate very few products")
print("   - Standard ML algorithms fail on sparse data")
print("   - Need specialized techniques like Matrix Factorization")

# ==================== 5. RATING DISTRIBUTION ====================
print("\n\n5️⃣ Rating Distribution")
print("-"*60)

rating_counts = ratings_df['rating'].value_counts().sort_index()
print(rating_counts)
print(f"\nMode (Most Common): {ratings_df['rating'].mode()[0]}")

# Check for rating bias
if ratings_df['rating'].mean() > 4.0:
    print("⚠️  Positive bias detected (avg > 4.0)")
    print("   Users tend to rate products they like")
elif ratings_df['rating'].mean() < 3.0:
    print("⚠️  Negative bias detected (avg < 3.0)")
else:
    print("✅ Relatively balanced rating distribution")

# ==================== 6. USER ACTIVITY ANALYSIS ====================
print("\n\n6️⃣ User Activity Analysis")
print("-"*60)

user_activity = ratings_df['user_id'].value_counts()

print(f"Most Active User: {user_activity.max()} ratings")
print(f"Least Active User: {user_activity.min()} ratings")
print(f"Average Ratings per User: {user_activity.mean():.2f}")
print(f"Median Ratings per User: {user_activity.median():.0f}")

# Cold start analysis
users_few_ratings = (user_activity < 5).sum()
users_many_ratings = (user_activity > 100).sum()

print(f"\n🔴 Users with <5 ratings: {users_few_ratings:,} ({users_few_ratings/len(user_activity)*100:.1f}%)")
print(f"   → These are 'cold start' users - hard to recommend for")
print(f"🟢 Users with >100 ratings: {users_many_ratings:,} ({users_many_ratings/len(user_activity)*100:.1f}%)")
print(f"   → These are 'power users' - reliable data source")

# ==================== 7. PRODUCT POPULARITY ANALYSIS ====================
print("\n\n7️⃣ Product Popularity Analysis")
print("-"*60)

product_popularity = ratings_df['product_id'].value_counts()

print(f"Most Popular Product: {product_popularity.max()} ratings")
print(f"Least Popular Product: {product_popularity.min()} ratings")
print(f"Average Ratings per Product: {product_popularity.mean():.2f}")
print(f"Median Ratings per Product: {product_popularity.median():.0f}")

# Cold start products
products_few_ratings = (product_popularity < 5).sum()
products_many_ratings = (product_popularity > 50).sum()

print(f"\n🔴 Products with <5 ratings: {products_few_ratings:,} ({products_few_ratings/len(product_popularity)*100:.1f}%)")
print(f"   → Niche/new products - need content-based approach")
print(f"🟢 Products with >50 ratings: {products_many_ratings:,} ({products_many_ratings/len(product_popularity)*100:.1f}%)")
print(f"   → Popular products - reliable collaborative filtering")

# ==================== 8. LOAD PRODUCT METADATA ====================
print("\n\n8️⃣ Loading Product Metadata")
print("-"*60)

try:
    products_df = pd.read_csv('data/raw/Amazon_Electronics.csv')
    print("✅ Product metadata loaded successfully!")
    print(f"Shape: {products_df.shape}")
    print(f"Columns: {list(products_df.columns)}")
    print("\nFirst 3 rows:")
    print(products_df.head(3))
    
    # Check for missing values
    print("\nMissing Values in Metadata:")
    print(products_df.isnull().sum())
    
except FileNotFoundError:
    print("⚠️  WARNING: electronics_product.csv not found")
    print("   Content-based filtering will be limited")
    products_df = None

# ==================== 9. DATASET OVERLAP ANALYSIS ====================
if products_df is not None:
    print("\n\n9️⃣ Dataset Overlap Analysis")
    print("-"*60)
    
    # Identify product ID column in metadata
    possible_id_columns = ['product_id', 'asin', 'id', 'item_id', 'productId']
    product_id_column = None
    
    for col in possible_id_columns:
        if col in products_df.columns:
            product_id_column = col
            break
    
    if product_id_column:
        print(f"✅ Product ID column identified: '{product_id_column}'")
        
        products_in_ratings = set(ratings_df['product_id'].unique())
        products_in_metadata = set(products_df[product_id_column].unique())
        
        overlap = products_in_ratings.intersection(products_in_metadata)
        
        print(f"\nProducts in Ratings: {len(products_in_ratings):,}")
        print(f"Products in Metadata: {len(products_in_metadata):,}")
        print(f"✅ Overlap: {len(overlap):,} products ({len(overlap)/len(products_in_ratings)*100:.1f}% coverage)")
        print(f"❌ Only in Ratings: {len(products_in_ratings - overlap):,}")
        print(f"❌ Only in Metadata: {len(products_in_metadata - overlap):,}")
        
        if len(overlap) / len(products_in_ratings) > 0.5:
            print("\n✅ Good overlap! Hybrid model will work well.")
        else:
            print("\n⚠️  Low overlap. Consider finding better matching metadata.")
    else:
        print("⚠️  Could not identify product ID column in metadata")
        print(f"Available columns: {products_df.columns.tolist()}")

# ==================== 10. VISUALIZATIONS ====================
print("\n\n🔟 Generating Visualizations...")
print("-"*60)

# Create output directory
os.makedirs('data/processed', exist_ok=True)

# Visualization 1: Rating Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

rating_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

ratings_df['rating'].hist(bins=20, ax=axes[1], color='coral', edgecolor='black')
axes[1].set_title('Rating Histogram', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Frequency')
axes[1].axvline(ratings_df['rating'].mean(), color='red', linestyle='--', 
                label=f'Mean: {ratings_df["rating"].mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/processed/rating_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: rating_distribution.png")

# Visualization 2: Long Tail Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(user_activity.values, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Number of Ratings per User', fontsize=12)
axes[0].set_ylabel('Number of Users (log scale)', fontsize=12)
axes[0].set_title('User Activity Distribution', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')

axes[1].hist(product_popularity.values, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Number of Ratings per Product', fontsize=12)
axes[1].set_ylabel('Number of Products (log scale)', fontsize=12)
axes[1].set_title('Product Popularity Distribution', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('data/processed/long_tail_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: long_tail_distribution.png")

# ==================== 11. SAVE SUMMARY STATISTICS ====================
print("\n\n1️⃣1️⃣ Saving Summary Statistics")
print("-"*60)

summary_stats = {
    'total_interactions': int(n_interactions),
    'unique_users': int(n_users),
    'unique_products': int(n_products),
    'sparsity': float(sparsity),
    'sparsity_percentage': float(sparsity * 100),
    'avg_rating': float(ratings_df['rating'].mean()),
    'median_rating': float(ratings_df['rating'].median()),
    'min_rating': float(ratings_df['rating'].min()),
    'max_rating': float(ratings_df['rating'].max()),
    'users_with_few_ratings': int(users_few_ratings),
    'users_with_many_ratings': int(users_many_ratings),
    'products_with_few_ratings': int(products_few_ratings),
    'products_with_many_ratings': int(products_many_ratings),
    'date_range': {
        'start': str(ratings_df['datetime'].min()),
        'end': str(ratings_df['datetime'].max())
    }
}

with open('data/processed/summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("✅ Summary statistics saved to: data/processed/summary_stats.json")

# ==================== 12. KEY TAKEAWAYS ====================
print("\n\n" + "="*60)
print("🎯 KEY TAKEAWAYS")
print("="*60)

print("\n1. SPARSITY:")
print(f"   Your matrix is {sparsity*100:.2f}% sparse")
print("   → Need Matrix Factorization techniques (SVD, NMF)")

print("\n2. COLD START PROBLEM:")
print(f"   {users_few_ratings:,} users with <5 ratings")
print(f"   {products_few_ratings:,} products with <5 ratings")
print("   → Need content-based fallback for these")

print("\n3. DATA QUALITY:")
if ratings_df.isnull().sum().sum() == 0:
    print("   ✅ No missing values - clean data!")
else:
    print("   ⚠️  Some missing values - need preprocessing")

print("\n4. NEXT STEPS:")
print("   → Data preprocessing (filtering, splitting)")
print("   → Build collaborative filtering models")
print("   → Add content-based features")
print("   → Create hybrid recommender")

print("\n" + "="*60)
print("🎉 PHASE 1 COMPLETE!")
print("="*60)
print("\nVisualization files saved in: data/processed/")
print("Review the plots to understand your data distribution.")
print("\nReady for Phase 2: Data Preprocessing 🚀")