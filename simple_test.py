#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple recommendation feature test script
"""

from book_recommender_simple import SimpleBookRecommender

def test_recommendations():
    """Test the recommendation features"""
    print("🎯 Testing recommendation features")
    print("=" * 50)
    
    # Initialize the recommendation system
    recommender = SimpleBookRecommender()
    
    # Load models
    if not recommender.load_models():
        print("❌ Failed to load models")
        return
    
    print("✅ Models loaded successfully")
    
    # Test user ID
    user_id = 30944
    print(f"\n👤 Test User ID: {user_id}")
    
    # Get personalized recommendations
    print("\n📚 Getting personalized recommendations...")
    try:
        recommendations = recommender.get_personalized_recommendations(user_id, 10)
        
        if len(recommendations) > 0:
            print(f"✅ Successfully retrieved {len(recommendations)} recommended books:")
            print("-" * 60)
            
            for i, book in recommendations.iterrows():
                print(f"{i+1:2d}. {book['title']}")
                print(f"    Author: {book['authors']}")
                print(f"    Rating: {book['average_rating']}")
                if 'predicted_rating' in book:
                    print(f"    Predicted Rating: {book['predicted_rating']:.2f}")
                if 'similarity_score' in book:
                    print(f"    Similarity Score: {book['similarity_score']:.3f}")
                print()
        else:
            print("❌ No recommendations found")
            
    except Exception as e:
        print(f"❌ Failed to get recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test popular book recommendations
    print("\n🔥 Getting popular book recommendations...")
    try:
        popular_books = recommender.get_popular_recommendations(5)
        
        if len(popular_books) > 0:
            print(f"✅ Successfully retrieved {len(popular_books)} popular books:")
            print("-" * 60)
            
            for i, book in popular_books.iterrows():
                print(f"{i+1:2d}. {book['title']}")
                print(f"    Author: {book['authors']}")
                print(f"    Rating: {book['average_rating']} ({book['ratings_count']} ratings)")
                print()
        else:
            print("❌ No popular books found")
            
    except Exception as e:
        print(f"❌ Failed to get popular books: {str(e)}")

if __name__ == "__main__":
    test_recommendations()
