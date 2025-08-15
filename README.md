# 📚 Personalized Book Recommendation System

A machine learning–based personalized book recommendation system supporting multiple recommendation algorithms, including collaborative filtering, content-based recommendation, and hybrid recommendation.

## 🎯 Features

* **Multiple Recommendation Algorithms**:

  * Content-Based Recommendation
  * Collaborative Filtering
  * Hybrid Recommendation
  * Popular Books Recommendation

* **Intelligent Recommendation Strategies**:

  * Cold-start handling for new users
  * Personalized recommendations
  * Real-time user interaction logging

* **Web Interface**:

  * User-friendly interface
  * RESTful API
  * Real-time recommendation display

## 📋 Requirements

* Python 3.10+
* Dependencies (see `requirement.txt`)

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

### 2. Train the models

```bash
python train_models.py
```

This will:

* Load and preprocess the data
* Train the content-based recommendation model
* Train the collaborative filtering model
* Train the hybrid recommendation model
* Save all models to the `models/` directory

### 3. Start the Web service

```bash
python app.py
```

Then visit in your browser: [http://localhost:5000](http://localhost:5000)

### 4. Run the demo

```bash
python demo.py
```

## 📁 Project Structure

```
Recommender System/
├── book_recommender.py      # Core recommendation system class
├── app.py                   # FastAPI web app
├── train_models.py          # Model training script
├── demo.py                  # Demo script
├── requirement.txt          # Dependencies list
├── README.md                # Project documentation
├── goodbooks-10k-master/    # Dataset folder
│   ├── books.csv            # Book information
│   ├── ratings.csv          # User ratings
│   ├── book_tags.csv        # Book-tag mapping
│   ├── tags.csv             # Tag information
│   └── to_read.csv          # Users' to-read list
└── models/                  # Trained models (auto-created)
```

## 🔧 Core Components

### BookRecommender Class

The main recommendation system class with the following methods:

* `load_data()`: Load data files
* `preprocess_data()`: Data preprocessing
* `build_content_based_model()`: Build content-based model
* `build_collaborative_model()`: Build collaborative filtering model
* `build_hybrid_model()`: Build hybrid model
* `get_content_recommendations()`: Get content-based recommendations
* `get_collaborative_recommendations()`: Get collaborative filtering recommendations
* `get_personalized_recommendations()`: Get personalized recommendations
* `get_popular_recommendations()`: Get popular book recommendations

### Web API Endpoints

* `GET /`: Main page
* `GET /api/content_recommendations/<book_id>`: Content-based recommendations
* `GET /api/personalized_recommendations/<user_id>`: Personalized recommendations
* `GET /api/popular_books`: Popular books
* `GET /api/system_info`: System info
* `GET /api/search_books`: Search books
* `POST /api/user_interaction`: Log user interactions

## 🎮 Usage Examples

### Basic Usage

```python
from book_recommender import BookRecommender

# Initialize
recommender = BookRecommender()

# Load trained models
recommender.load_models()

# Get popular books
popular_books = recommender.get_popular_recommendations(10)

# Get content-based recommendations
content_recs = recommender.get_content_recommendations(book_id=1, n_recommendations=5)

# Get personalized recommendations
personalized_recs = recommender.get_personalized_recommendations(user_id=1, n_recommendations=10)
```

### Web Interface Usage

1. Start the web service and visit [http://localhost:5000](http://localhost:5000)
2. In "Book-based Recommendation," enter a book ID to get similar books
3. In "Personalized Recommendation," enter a user ID to get personalized recommendations
4. Click "Get Popular Books" to see the most popular books

## 📊 Dataset

Using the Goodreads 10K dataset, which includes:

* **books.csv**: Basic info for 10,000 books
* **ratings.csv**: User ratings
* **book\_tags.csv**: Book-tag relationships
* **tags.csv**: Tag info
* **to\_read.csv**: Users' to-read lists

## 🔍 Recommendation Algorithms

### 1. Content-Based

* TF-IDF vectorization of title, author, and tags
* Cosine similarity between books
* Recommend based on similarity

### 2. Collaborative Filtering

* SVD (Singular Value Decomposition)
* Based on user-book rating matrix
* Predict ratings for unseen books

### 3. Hybrid Recommendation

* Combine user and book features
* LightGBM gradient boosting
* Merge multiple strategies

### 4. Popular Recommendation

* Based on book popularity score
* Consider average rating and rating count
* Suitable for cold-start users

## ⚙️ Config Options

Adjustable in `book_recommender.py`:

* TF-IDF features: `max_features=5000`
* SVD factors: `n_factors=100`
* LightGBM: `n_estimators=100`, `learning_rate=0.1`
* Default recommendation count: 10

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**

   * Make sure to run `python train_models.py`
   * Check that `models/` exists

2. **Out of memory**

   * Reduce TF-IDF `max_features`
   * Use a subset of the dataset

3. **Long training time**

   * Use sample data for quick tests
   * Reduce model parameters

### Performance Tips

* Use smaller datasets for dev/testing
* Tune parameters for balance
* Consider faster algorithm implementations

## 📈 Model Evaluation

Automatically evaluates:

* **Collaborative Filtering**: RMSE & MAE
* **Hybrid**: RMSE & MAE
* **Content-Based**: Similarity matrix

## 🔄 Model Updating

```python
# Retrain all models
recommender.train_all_models()

# Save models
recommender.save_models()
```

## 📝 Roadmap

* [ ] Add more algorithms (e.g., deep learning)
* [ ] Real-time recommendation updates
* [ ] Explainable recommendations
* [ ] More data sources
* [ ] Improved web UI
* [ ] User authentication

## 🤝 Contributing

Feel free to submit Issues and Pull Requests.

## 📄 License

MIT License.

## 🙏 Acknowledgements

* Goodreads dataset providers
* Contributors to open-source ML libraries
* Recommender systems research community