#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debugging script for the recommendation feature
"""

from book_recommender_simple import SimpleBookRecommender
import pandas as pd

def debug_recommendation():
    """Debug the recommendation feature"""
    print("🔍 Starting recommendation feature debugging...")
    
    # Initialize recommender system
    recommender = SimpleBookRecommender()
    
    # Check model loading
    print("\n1. Checking model loading...")
    if recommender.load_models():
        print("✅ Model loaded successfully")
    else:
        print("❌ Model loading failed")
        return
    
    # Check data
    print("\n2. Checking data...")
    if recommender.books_df is not None:
        print(f"✅ Books data: {len(recommender.books_df)} books")
    else:
        print("❌ Books data not loaded")
        return
    
    if recommender.ratings_df is not None:
        print(f"✅ Ratings data: {len(recommender.ratings_df)} records")
    else:
        print("❌ Ratings data not loaded")
        return
    
    # Test with user ID 30944
    user_id = 30944
    print(f"\n3. Testing user {user_id}...")
    
    # Check if user exists
    user_ratings = recommender.ratings_df[recommender.ratings_df['user_id'] == user_id]
    print(f"User {user_id} has {len(user_ratings)} ratings")
    
    if len(user_ratings) == 0:
        print("❌ User does not exist, trying another user ID...")
        # Pick an existing user
        sample_user = recommender.ratings_df['user_id'].iloc[0]
        user_id = sample_user
        print(f"Using user ID: {user_id}")
        user_ratings = recommender.ratings_df[recommender.ratings_df['user_id'] == user_id]
    
    # Test collaborative filtering recommendations
    print(f"\n4. Testing collaborative filtering recommendations...")
    try:
        cf_recs = recommender.get_collaborative_recommendations(user_id, 5)
        print(f"Collaborative filtering results: {len(cf_recs)} books")
        if len(cf_recs) > 0:
            print("Top 3 recommended books:")
            for i, book in cf_recs.head(3).iterrows():
                print(f"  - {book['title']} (Predicted rating: {book['predicted_rating']:.2f})")
    except Exception as e:
        print(f"❌ Collaborative filtering failed: {str(e)}")
    
    # Test content-based recommendations
    print(f"\n5. Testing content-based recommendations...")
    try:
        # Get the user's favorite book
        user_favorite = user_ratings.loc[user_ratings['rating'].idxmax(), 'book_id']
        print(f"User's favorite book ID: {user_favorite}")
        
        content_recs = recommender.get_content_recommendations(user_favorite, 5)
        print(f"Content-based recommendation results: {len(content_recs)} books")
        if len(content_recs) > 0:
            print("Top 3 recommended books:")
            for i, book in content_recs.head(3).iterrows():
                print(f"  - {book['title']} (Similarity score: {book['similarity_score']:.3f})")
    except Exception as e:
        print(f"❌ Content-based recommendation failed: {str(e)}")
    
    # Test personalized recommendations
    print(f"\n6. Testing personalized recommendations...")
    try:
        personalized_recs = recommender.get_personalized_recommendations(user_id, 10)
        print(f"Personalized recommendation results: {len(personalized_recs)} books")
        if len(personalized_recs) > 0:
            print("Top 5 recommended books:")
            for i, book in personalized_recs.head(5).iterrows():
                print(f"  - {book['title']} (Author: {book['authors']})")
        else:
            print("❌ No personalized recommendation results")
    except Exception as e:
        print(f"❌ Personalized recommendation failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test popular recommendations
    print(f"\n7. Testing popular recommendations...")
    try:
        popular_recs = recommender.get_popular_recommendations(5)
        print(f"Popular recommendation results: {len(popular_recs)} books")
        if len(popular_recs) > 0:
            print("Top 3 popular books:")
            for i, book in popular_recs.head(3).iterrows():
                print(f"  - {book['title']} (Rating: {book['average_rating']})")
    except Exception as e:
        print(f"❌ Popular recommendation failed: {str(e)}")

if __name__ == "__main__":
    debug_recommendation()
