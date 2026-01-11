
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Initialize the API
app = FastAPI(title="Ecommerce AI Recommender")

# 2. Load the model and data
with open('ecommerce_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

# Extract needed parts
U, sigma, Vt = artifacts['U'], artifacts['sigma'], artifacts['Vt']
product_profiles = artifacts['product_profiles']
user_map, item_map = artifacts['user_map'], artifacts['item_map']
df = pd.read_csv('ecommerce_recommendation_dataset.csv')
df['unique_item_id'] = df['category'] + "_" + df['product_id'].astype(str)

@app.get("/")
def home():
    return {"message": "Ecommerce Recommendation API is Live!"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    # Logic for recommendations
    if user_id not in user_map:
        # Fallback to popularity
        recs = df.sort_values('product_popularity', ascending=False)['unique_item_id'].unique()[:n].tolist()
        return {"status": "success", "type": "popularity", "recommendations": recs}

    user_idx = user_map[user_id]
    user_history = df[df['user_id'] == user_id]
    last_item = user_history['unique_item_id'].iloc[-1]
    
    # Simple similarity check
    target_features = product_profiles.loc[[last_item]].values
    sims = cosine_similarity(target_features, product_profiles.values).flatten()
    
    top_indices = sims.argsort()[-(n+1):][::-1]
    recs = [r for r in product_profiles.index[top_indices].tolist() if r != last_item][:n]
    
    return {
        "user_id": user_id,
        "last_item_viewed": last_item,
        "recommendations": recs
    }
