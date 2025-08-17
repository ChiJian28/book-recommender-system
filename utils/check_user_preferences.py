#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to check user preferences
"""

from models.book_recommender_simple import SimpleBookRecommender
import pandas as pd

def check_user_preferences():
    """Check the preferences of user 30944"""
    print("🔍 Checking reading preferences for user 30944")
    print("=" * 60)
    
    # Initialize the recommender system
    recommender = SimpleBookRecommender()
    
    # Load models
    if not recommender.load_models():
        print("❌ Failed to load models")
        return
    
    user_id = 30944
    
    # Retrieve user's rating records
    user_ratings = recommender.ratings_df[recommender.ratings_df['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        print(f"❌ No rating records found for user {user_id}")
        return
    
    # Merge with book information
    user_books = user_ratings.merge(recommender.books_df, on='book_id')
    
    print(f"👤 Statistics for user {user_id}:")
    print(f"   Total ratings: {len(user_books)}")
    print(f"   Average rating: {user_books['rating'].mean():.2f}")
    print(f"   Rating standard deviation: {user_books['rating'].std():.2f}")
    
    # Sort by rating
    user_books_sorted = user_books.sort_values('rating', ascending=False)
    
    print(f"\n📖 User's favorite books (5-star rating):")
    print("-" * 60)
    top_books = user_books_sorted[user_books_sorted['rating'] == 5].head(10)
    for i, book in top_books.iterrows():
        print(f"  {book['title']} (Author: {book['authors']})")
    
    print(f"\n📖 User's highly rated books (4-star rating):")
    print("-" * 60)
    good_books = user_books_sorted[user_books_sorted['rating'] == 4].head(10)
    for i, book in good_books.iterrows():
        print(f"  {book['title']} (Author: {book['authors']})")
    
    print(f"\n📖 User's moderately rated books (3-star rating):")
    print("-" * 60)
    medium_books = user_books_sorted[user_books_sorted['rating'] == 3].head(10)
    for i, book in medium_books.iterrows():
        print(f"  {book['title']} (Author: {book['authors']})")
    
    print(f"\n📖 User's disliked books (1-2 star rating):")
    print("-" * 60)
    low_books = user_books_sorted[user_books_sorted['rating'] <= 2].head(10)
    for i, book in low_books.iterrows():
        print(f"  {book['title']} (Author: {book['authors']}) - Rating: {book['rating']}")
    
    # Analyze favorite authors
    print(f"\n👨‍💼 User's favorite authors (based on 5-star ratings):")
    print("-" * 60)
    top_authors = top_books['authors'].value_counts().head(5)
    for author, count in top_authors.items():
        print(f"  {author}: {count} 5-star books")
    
    # Analyze preferred book genres (based on title keywords)
    print(f"\n📚 User's preferred book genres analysis:")
    print("-" * 60)
    
    # Count books with specific keywords
    fantasy_keywords = ['harry potter', 'lord of the rings', 'game of thrones', 'fantasy', 'magic']
    classic_keywords = ['pride and prejudice', 'jane eyre', 'wuthering heights', 'classic']
    modern_keywords = ['hunger games', 'divergent', 'twilight', 'young adult']
    
    fantasy_count = sum(1 for title in user_books_sorted['title'].str.lower() 
                       if any(keyword in title for keyword in fantasy_keywords))
    classic_count = sum(1 for title in user_books_sorted['title'].str.lower() 
                       if any(keyword in title for keyword in classic_keywords))
    modern_count = sum(1 for title in user_books_sorted['title'].str.lower() 
                      if any(keyword in title for keyword in modern_keywords))
    
    print(f"  Fantasy/Magic: {fantasy_count} books")
    print(f"  Classic Literature: {classic_count} books")
    print(f"  Modern Young Adult: {modern_count} books")
    
    print(f"\n🎯 Recommendation rationale:")
    print("-" * 60)
    print("Based on user 30944's reading preferences, the system recommends the following types of books:")
    print("1. Fantasy/Magic books (e.g., Harry Potter series)")
    print("2. Classic literature (e.g., Pride and Prejudice)")
    print("3. Modern young adult novels (e.g., The Hunger Games)")
    print("4. Historical novels and war-themed books")
    print("5. Social issue novels")

if __name__ == "__main__":
    check_user_preferences()