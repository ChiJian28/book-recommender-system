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
 
## 👀 Demo
[Watch the demo video](https://github.com/user-attachments/assets/5b13d83b-9194-4da2-9dd3-f5420fdc202f)

- **Book-Based:** Searching Book ID 20 (*The Hunger Games*) shows similar books: Hunger Games–related titles.  
- **Personalized:** User ID 150 (loves Sci-Fi) gets recommendations: *Harry Potter*, *The Wise Man’s Fear*, etc.

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

## 📊 Data Pipeline & Model Training

### Data Collection

The system utilizes the **Goodreads 10K dataset**, a comprehensive collection of book-related data that serves as the foundation for our recommendation algorithms. This dataset provides rich information about books, user interactions, and content metadata.

**Dataset Components**:
- **Book Metadata**: Contains detailed information about 10,000 books including titles, authors, publication years, average ratings, and rating counts
- **User Interactions**: Records of user-book rating interactions, capturing individual user preferences and behaviors
- **Content Tags**: Book-tag relationships that enable content-based recommendation by categorizing books into genres, themes, and topics
- **Tag Information**: Descriptive metadata for tags, providing context for content classification
- **Reading Lists**: Users' to-read lists that offer insights into future reading intentions and preferences

**Data Scale and Characteristics**:
- **Volume**: Approximately 10,000 books, 53,000 users, and 6 million ratings
- **Sparsity**: Rating matrix sparsity of approximately 1.1%, which is typical for recommendation systems and presents both challenges and opportunities for algorithm design
- **Quality**: High-quality, real-world data that reflects genuine user behavior and preferences

### Data Preprocessing

#### Data Cleaning and Validation

The preprocessing pipeline begins with comprehensive data cleaning to ensure data quality and consistency. Missing values in book metadata are handled systematically - unknown authors are marked appropriately, missing titles are flagged, and language codes are standardized. Invalid ratings outside the 1-5 scale are removed to maintain data integrity.

The system implements robust data validation checks to ensure the dataset meets quality standards before proceeding to feature engineering. These checks verify data completeness, rating validity, and referential integrity between different data tables.

#### Feature Engineering

The feature engineering process transforms raw data into meaningful representations that capture the essential characteristics needed for effective recommendation generation.

**Text Feature Creation**: For content-based recommendation, the system combines book titles, author names, and associated tags into comprehensive text features. This unified representation enables the algorithm to understand book content and find similar items based on textual characteristics.

**Popularity Scoring**: A sophisticated popularity metric is calculated that balances average rating with rating count using a logarithmic transformation. This approach prevents highly-rated books with few ratings from dominating recommendations while ensuring popular books receive appropriate consideration.

**Temporal Features**: Publication year information is extracted and normalized to capture temporal patterns in user preferences and book popularity trends.

#### Data Transformation

The preprocessing pipeline transforms the raw dataset into structured formats optimized for machine learning algorithms. User-book interactions are organized into matrices suitable for collaborative filtering, while text features are prepared for natural language processing techniques.

### Model Training Pipeline

#### Content-Based Recommendation Training

The content-based recommendation system employs natural language processing techniques to understand book similarities based on textual content. The training process begins with text vectorization using TF-IDF (Term Frequency-Inverse Document Frequency) techniques, which convert textual descriptions into numerical representations that capture the importance of different words and phrases.

The system then computes similarity matrices that quantify the relationships between all book pairs based on their content characteristics. This pre-computed similarity matrix enables fast retrieval of similar books during recommendation generation, significantly improving system performance.

**Training Process**:
1. **Text Analysis**: Extract and process textual features from book metadata
2. **Vectorization**: Convert text to numerical representations using TF-IDF
3. **Similarity Computation**: Calculate pairwise similarities between all books
4. **Matrix Optimization**: Store similarity data in efficient data structures for rapid access

#### Collaborative Filtering Training

The collaborative filtering approach learns user preferences by analyzing patterns in the user-book rating matrix. The training process addresses the inherent sparsity of rating data through sophisticated dimensionality reduction techniques.

**Matrix Factorization**: The system employs Singular Value Decomposition (SVD) to extract latent features from the sparse rating matrix. This technique identifies underlying patterns in user preferences and book characteristics that may not be immediately apparent from the raw data.

**Feature Engineering**: Comprehensive user and book features are engineered to enhance prediction accuracy. User features include average rating patterns, rating consistency, and reading diversity, while book features incorporate popularity metrics, publication information, and content characteristics.

**Model Training**: A Random Forest regressor is trained to predict user ratings for unrated books. This ensemble method provides robust predictions while handling the non-linear relationships present in user preference data.

**Training Process**:
1. **Matrix Construction**: Organize user-book interactions into a structured matrix format
2. **Dimensionality Reduction**: Apply SVD to extract latent features and reduce sparsity
3. **Feature Enhancement**: Create comprehensive user and book behavior features
4. **Model Optimization**: Train ensemble models for accurate rating prediction

#### Model Evaluation and Validation

The training pipeline includes comprehensive evaluation procedures to ensure model quality and performance. The collaborative filtering model is evaluated using standard regression metrics including Root Mean Square Error (RMSE) and Mean Absolute Error (MAE).

**Performance Targets**: The system aims for RMSE values below 1.0 and MAE values below 0.8, which represent good prediction accuracy for rating prediction tasks. These targets are based on industry standards and empirical testing with the dataset.

**Validation Strategy**: Models are validated using holdout datasets to ensure generalization performance and prevent overfitting. Cross-validation techniques are employed where appropriate to provide robust performance estimates.

### Model Persistence and Deployment

#### Model Storage and Management

Trained models are systematically saved to enable efficient deployment and reuse. The persistence system stores both the trained models and the processed data required for recommendation generation.

**Stored Components**:
- **Content Similarity Matrix**: Pre-computed book similarity data for fast content-based recommendations
- **TF-IDF Vectorizer**: Fitted text processing model for consistent feature extraction
- **Collaborative Filtering Model**: Trained Random Forest model for rating prediction
- **SVD Model**: Dimensionality reduction model for latent feature extraction
- **Processed Data**: Cleaned and feature-engineered datasets for efficient access

#### Performance Optimization

The system implements several optimization strategies to ensure efficient training and deployment:

**Memory Management**: TF-IDF features are limited to 3,000 dimensions to balance representation quality with memory efficiency. SVD dimensionality reduction reduces the feature space to 50 components while preserving essential information.

**Computational Efficiency**: Similarity matrices are pre-computed during training to enable constant-time similarity queries during recommendation generation. Vectorized operations are employed throughout the pipeline to maximize computational efficiency.

**Storage Optimization**: Models are compressed and stored in efficient formats to minimize storage requirements while maintaining fast loading times.

### Training Performance Characteristics

**Resource Requirements**: The complete training pipeline typically requires 2-5 minutes of processing time, depending on hardware specifications. Memory usage ranges from 2-4 GB RAM, making the system accessible on standard development machines.

**Model Size**: The complete model ensemble requires approximately 500MB of storage, including all similarity matrices, trained models, and processed data. This compact representation enables efficient deployment and updates.

**Scalability Considerations**: The modular design allows for easy scaling to larger datasets by adjusting feature dimensions and model parameters. The system can be extended to handle millions of books and users with appropriate hardware resources.

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

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Summary

👉 If you found this project helpful, please ⭐ it and share it with others!
