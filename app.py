from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from book_recommender_simple import SimpleBookRecommender as BookRecommender
import pandas as pd
import os
import uvicorn

app = FastAPI(
    title="Book Recommendation System",
    description="Personalized book recommendation system based on machine learning",
    version="1.0.0"
)

# 初始化推荐系统
recommender = BookRecommender()

# Check if trained models exist
if not recommender.load_models():
    print("No trained models found, please run train_models.py first")

# Pydantic models for requests and responses
class UserInteraction(BaseModel):
    user_id: int
    book_id: int
    interaction_type: str = "view"

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

class SystemInfoResponse(BaseModel):
    success: bool
    total_books: Optional[int] = None
    total_users: Optional[int] = None
    total_ratings: Optional[int] = None
    message: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Book Recommendation System</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            input, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 3px; }
            button { background-color: #007bff; color: white; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .book-item { margin: 10px 0; padding: 10px; border: 1px solid #eee; border-radius: 3px; }
            .book-title { font-weight: bold; color: #333; }
            .book-author { color: #666; font-style: italic; }
            .book-rating { color: #007bff; }
            .api-docs { text-align: center; margin: 20px 0; }
            .api-docs a { color: #007bff; text-decoration: none; }
            .api-docs a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Personalized Book Recommendation System</h1>
            <div class="api-docs">
                <a href="/docs" target="_blank">📖 View API Documentation</a> | 
                <a href="/redoc" target="_blank">📋 View ReDoc Documentation</a>
            </div>
            
            <div class="section">
                <h3>🔍 Content-Based Recommendations</h3>
                <p>Enter a book ID to get similar book recommendations</p>
                <input type="number" id="bookId" placeholder="Enter Book ID" min="1">
                <button onclick="getContentRecommendations()">Get Recommendations</button>
                <div id="contentResults"></div>
            </div>
            
            <div class="section">
                <h3>👤 Personalized Recommendations</h3>
                <p>Enter a user ID to get personalized book recommendations</p>
                <input type="number" id="userId" placeholder="Enter User ID" min="1">
                <button onclick="getPersonalizedRecommendations()">Get Recommendations</button>
                <div id="personalizedResults"></div>
            </div>
            
            <div class="section">
                <h3>🔥 Popular Books</h3>
                <p>Get the most popular books currently</p>
                <button onclick="getPopularBooks()">Get Popular Books</button>
                <div id="popularResults"></div>
            </div>
            
            <div class="section">
                <h3>📊 System Information</h3>
                <p>Total Books: <span id="totalBooks">-</span></p>
                <p>Total Users: <span id="totalUsers">-</span></p>
                <p>Total Ratings: <span id="totalRatings">-</span></p>
                <button onclick="getSystemInfo()">Refresh Info</button>
            </div>
        </div>
        
        <script>
            function getContentRecommendations() {
                const bookId = document.getElementById('bookId').value;
                if (!bookId) {
                    alert('Please enter a book ID');
                    return;
                }
                
                // Show loading state
                const resultsDiv = document.getElementById('contentResults');
                resultsDiv.innerHTML = '<p style="color: blue;">Loading recommendations, please wait...</p>';
                
                fetch(`/api/content_recommendations/${bookId}`)
                    .then(response => {
                        console.log('Response status:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('Response data:', data);
                        if (data.success) {
                            let html = '<h4>Recommended Books:</h4>';
                            data.recommendations.forEach(book => {
                                const similarity = book.similarity_score !== null ? book.similarity_score.toFixed(3) : 'N/A';
                                html += `
                                    <div class="book-item">
                                        <div class="book-title">${book.title}</div>
                                        <div class="book-author">Author: ${book.authors}</div>
                                        <div class="book-rating">Rating: ${book.average_rating} | Similarity: ${similarity}</div>
                                    </div>
                                `;
                            });
                            resultsDiv.innerHTML = html;
                        } else {
                            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                        }
                    })
                    .catch(error => {
                        console.error('Fetch error:', error);
                        resultsDiv.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
                    });
            }
            
            function getPersonalizedRecommendations() {
                const userId = document.getElementById('userId').value;
                if (!userId) {
                    alert('Please enter a user ID');
                    return;
                }
                
                // Show loading state
                const resultsDiv = document.getElementById('personalizedResults');
                resultsDiv.innerHTML = '<p style="color: blue;">Loading recommendations, please wait...</p>';
                
                fetch(`/api/personalized_recommendations/${userId}`)
                    .then(response => {
                        console.log('Response status:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('Response data:', data);
                        if (data.success) {
                            let html = '<h4>Personalized Recommendations:</h4>';
                            data.recommendations.forEach(book => {
                                html += `
                                    <div class="book-item">
                                        <div class="book-title">${book.title}</div>
                                        <div class="book-author">Author: ${book.authors}</div>
                                        <div class="book-rating">Rating: ${book.average_rating}</div>
                                    </div>
                                `;
                            });
                            resultsDiv.innerHTML = html;
                        } else {
                            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                        }
                    })
                    .catch(error => {
                        console.error('Fetch error:', error);
                        resultsDiv.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
                    });
            }
            
            function getPopularBooks() {
                // Show loading state
                const resultsDiv = document.getElementById('popularResults');
                resultsDiv.innerHTML = '<p style="color: blue;">Loading popular books, please wait...</p>';
                
                fetch('/api/popular_books')
                    .then(response => {
                        console.log('Response status:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('Response data:', data);
                        if (data.success) {
                            let html = '<h4>Popular Books:</h4>';
                            data.recommendations.forEach(book => {
                                html += `
                                    <div class="book-item">
                                        <div class="book-title">${book.title}</div>
                                        <div class="book-author">Author: ${book.authors}</div>
                                        <div class="book-rating">Rating: ${book.average_rating} | Rating Count: ${book.ratings_count}</div>
                                    </div>
                                `;
                            });
                            resultsDiv.innerHTML = html;
                        } else {
                            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                        }
                    })
                    .catch(error => {
                        console.error('Fetch error:', error);
                        resultsDiv.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
                    });
            }
            
            function getSystemInfo() {
                fetch('/api/system_info')
                    .then(response => {
                        console.log('System info response status:', response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log('System info data:', data);
                        if (data.success) {
                            document.getElementById('totalBooks').textContent = data.total_books;
                            document.getElementById('totalUsers').textContent = data.total_users;
                            document.getElementById('totalRatings').textContent = data.total_ratings;
                        } else {
                            console.error('System info error:', data.message);
                        }
                    })
                    .catch(error => {
                        console.error('System info fetch error:', error);
                    });
            }
            
            // Get system info when page loads
            window.onload = function() {
                console.log('Page loaded, getting system info...');
                getSystemInfo();
            };
        </script>
    </body>
    </html>
    """
    return html

@app.get("/api/content_recommendations/{book_id}", response_model=RecommendationResponse)
async def content_recommendations(book_id: int):
    """Content-based recommendation API"""
    try:
        recommendations = recommender.get_content_recommendations(book_id, 10)
        if len(recommendations) > 0:
            # Handle NaN values by converting them to None
            recommendations_dict = recommendations.to_dict('records')
            for book in recommendations_dict:
                for key, value in book.items():
                    if pd.isna(value):
                        book[key] = None
            
            return RecommendationResponse(
                success=True,
                recommendations=recommendations_dict
            )
        else:
            return RecommendationResponse(
                success=False,
                message=f'No recommendations found for book ID {book_id}'
            )
    except Exception as e:
        return RecommendationResponse(
            success=False,
            message=f'Failed to get recommendations: {str(e)}'
        )

@app.get("/api/personalized_recommendations/{user_id}", response_model=RecommendationResponse)
async def personalized_recommendations(user_id: int):
    """Personalized recommendation API"""
    try:
        print(f"Getting personalized recommendations for user {user_id}...")
        recommendations = recommender.get_personalized_recommendations(user_id, 10)
        print(f"Found {len(recommendations)} recommended books")
        
        if len(recommendations) > 0:
            # Handle NaN values by converting them to None
            recommendations_dict = recommendations.to_dict('records')
            for book in recommendations_dict:
                for key, value in book.items():
                    if pd.isna(value):
                        book[key] = None
            
            return RecommendationResponse(
                success=True,
                recommendations=recommendations_dict
            )
        else:
            return RecommendationResponse(
                success=False,
                message=f'No recommendations found for user ID {user_id}'
            )
    except Exception as e:
        print(f"Personalized recommendation API error: {str(e)}")
        import traceback
        traceback.print_exc()
        return RecommendationResponse(
            success=False,
            message=f'Failed to get recommendations: {str(e)}'
        )

@app.get("/api/popular_books", response_model=RecommendationResponse)
async def popular_books():
    """Popular books recommendation API"""
    try:
        recommendations = recommender.get_popular_recommendations(10)
        # Handle NaN values by converting them to None
        recommendations_dict = recommendations.to_dict('records')
        for book in recommendations_dict:
            for key, value in book.items():
                if pd.isna(value):
                    book[key] = None
        
        return RecommendationResponse(
            success=True,
            recommendations=recommendations_dict
        )
    except Exception as e:
        return RecommendationResponse(
            success=False,
            message=f'Failed to get popular books: {str(e)}'
        )

@app.get("/api/system_info", response_model=SystemInfoResponse)
async def system_info():
    """System information API"""
    try:
        if recommender.books_df is not None and recommender.ratings_df is not None:
            return SystemInfoResponse(
                success=True,
                total_books=len(recommender.books_df),
                total_users=recommender.ratings_df['user_id'].nunique(),
                total_ratings=len(recommender.ratings_df)
            )
        else:
            return SystemInfoResponse(
                success=False,
                message='Data not loaded'
            )
    except Exception as e:
        return SystemInfoResponse(
            success=False,
            message=f'Failed to get system info: {str(e)}'
        )

@app.get("/api/search_books")
async def search_books(q: str):
    """Search books API"""
    if not q:
        return JSONResponse({
            'success': False, 
            'message': 'Please enter search keywords'
        })
    
    try:
        if recommender.books_df is not None:
            # Simple text search
            mask = (
                recommender.books_df['title'].str.lower().str.contains(q.lower()) |
                recommender.books_df['authors'].str.lower().str.contains(q.lower())
            )
            results = recommender.books_df[mask].head(10)[
                ['book_id', 'title', 'authors', 'average_rating']
            ]
            
            return JSONResponse({
                'success': True,
                'books': results.to_dict('records')
            })
        else:
            return JSONResponse({
                'success': False,
                'message': 'Data not loaded'
            })
    except Exception as e:
        return JSONResponse({
            'success': False,
            'message': f'Search failed: {str(e)}'
        })

@app.post("/api/user_interaction")
async def record_interaction(interaction: UserInteraction):
    """Record user interaction API"""
    try:
        recommender.record_user_interaction(
            interaction.user_id, 
            interaction.book_id, 
            interaction.interaction_type
        )
        
        return JSONResponse({
            'success': True,
            'message': 'Interaction recorded successfully'
        })
    except Exception as e:
        return JSONResponse({
            'success': False,
            'message': f'Failed to record interaction: {str(e)}'
        })

if __name__ == "__main__":
    print("Starting Book Recommendation System Web Service...")
    print("Visit http://localhost:8080 to view the Web interface")
    print("Visit http://localhost:8080/docs to view API documentation")
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)
