#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User ID Finder Script
"""

import pandas as pd

def find_user_ids():
    """Find some user IDs for testing"""
    print("🔍 Searching for user IDs...")
    
    # Read ratings data
    ratings_df = pd.read_csv("goodbooks-10k-master/ratings.csv")
    
    # Count number of ratings per user
    user_rating_counts = ratings_df.groupby('user_id').size().reset_index(name='rating_count')
    user_rating_counts = user_rating_counts.sort_values('rating_count', ascending=False)
    
    # Calculate average rating per user
    user_avg_ratings = ratings_df.groupby('user_id')['rating'].mean().reset_index(name='avg_rating')
    
    # Merge user statistics
    user_stats = user_rating_counts.merge(user_avg_ratings, on='user_id')
    
    print(f"\n📊 User statistics:")
    print(f"- Total users: {len(user_stats)}")
    print(f"- Total ratings: {len(ratings_df)}")
    print(f"- Average ratings per user: {len(ratings_df) / len(user_stats):.1f}")
    
    # Show users with the most ratings
    print(f"\n🏆 Top users by number of ratings (good for testing):")
    print("=" * 60)
    
    top_users = user_stats.head(10)
    for _, user in top_users.iterrows():
        print(f"👤 User ID: {user['user_id']}")
        print(f"   Number of ratings: {user['rating_count']}")
        print(f"   Average rating: {user['avg_rating']:.2f}")
        print("-" * 60)
    
    # Show high-quality users (moderate number of ratings, high average score)
    print(f"\n⭐ High-quality users (moderate ratings count, high average score):")
    print("=" * 60)
    
    # Filter: rating count between 50-500, average rating > 3.5
    quality_users = user_stats[
        (user_stats['rating_count'] >= 50) & 
        (user_stats['rating_count'] <= 500) & 
        (user_stats['avg_rating'] > 3.5)
    ].sort_values('avg_rating', ascending=False)
    
    for _, user in quality_users.head(10).iterrows():
        print(f"👤 User ID: {user['user_id']}")
        print(f"   Number of ratings: {user['rating_count']}")
        print(f"   Average rating: {user['avg_rating']:.2f}")
        print("-" * 60)
    
    # Show a random sample of users
    print(f"\n🎲 Random user sample:")
    print("=" * 60)
    
    random_users = user_stats.sample(min(10, len(user_stats)))
    for _, user in random_users.iterrows():
        print(f"👤 User ID: {user['user_id']}")
        print(f"   Number of ratings: {user['rating_count']}")
        print(f"   Average rating: {user['avg_rating']:.2f}")
        print("-" * 60)
    
    return user_stats

def get_user_book_preferences(user_id):
    """Get book preferences for a specific user"""
    print(f"\n📚 Book preferences for user {user_id}:")
    print("=" * 60)
    
    # Read data
    ratings_df = pd.read_csv("goodbooks-10k-master/ratings.csv")
    books_df = pd.read_csv("goodbooks-10k-master/books.csv")
    
    # Get ratings from this user
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        print(f"❌ No ratings found for user {user_id}")
        return
    
    # Merge with book information
    user_books = user_ratings.merge(books_df, on='book_id')
    
    # Sort by rating
    user_books = user_books.sort_values('rating', ascending=False)
    
    print(f"User {user_id} has rated {len(user_books)} books")
    print(f"Average rating: {user_books['rating'].mean():.2f}")
    
    print(f"\n📖 Favorite books (rating = 5):")
    top_books = user_books[user_books['rating'] == 5].head(5)
    for _, book in top_books.iterrows():
        print(f"  - {book['title']} (Author: {book['authors']})")
    
    print(f"\n📖 Highly rated books (rating = 4):")
    good_books = user_books[user_books['rating'] == 4].head(5)
    for _, book in good_books.iterrows():
        print(f"  - {book['title']} (Author: {book['authors']})")

if __name__ == "__main__":
    print("🔍 User ID Finder Tool")
    print("=" * 50)
    
    # Find user IDs
    user_stats = find_user_ids()
    
    # Show detailed preferences for a few users
    print(f"\n🔍 Viewing specific users' book preferences:")
    sample_user_ids = user_stats.head(3)['user_id'].tolist()
    
    for user_id in sample_user_ids:
        get_user_book_preferences(user_id)
        print("\n" + "="*80 + "\n")
    
    print("🎉 Search complete!")
    print("\n💡 Testing suggestions:")
    print("1. Use the above user IDs for personalized recommendation testing")
    print("2. Choose users with a moderate number of ratings (50-500)")
    print("3. Choose users with high average ratings (> 3.5)")
    print("4. Run 'python app.py' to start the web interface for testing")
