import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_RATINGS_PATH = "data/raw/ratings_Electronics.csv"
RAW_PRODUCTS_PATH = "data/raw/Amazon_Electronics.csv"

def load_data():
    ratings = pd.read_csv(RAW_RATINGS_PATH, names=['user_id', 'product_id', 'rating', 'timestamp'])
    products = pd.read_csv(RAW_PRODUCTS_PATH)
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    return ratings, products

def explore_data(ratings, products):
    print("Ratings DataFrame shape:", ratings.shape)
    print("Products DataFrame shape:", products.shape)
    print("\nRatings head:\n", ratings.head())
    print("\nProducts head:\n", products.head())

    print("\nUnique users:", ratings['user_id'].nunique())
    print("Unique products:", ratings['product_id'].nunique())

    plt.figure(figsize=(6, 4))
    sns.countplot(x='rating', data=ratings)
    plt.title("Rating Distribution")
    plt.savefig("data/processed/rating_distribution.png")
    plt.close()

    print("\nRating distribution plot saved in data/processed/")

def main():
    os.makedirs("data/processed", exist_ok=True)
    ratings, products = load_data()
    explore_data(ratings, products)
    print("\nPhase 1 Complete: Data exploration done.")

if __name__ == "__main__":
    main()
