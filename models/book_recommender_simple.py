import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleBookRecommender:
    """
    简化版个性化图书推荐系统
    支持基于内容的推荐、协同过滤和热门推荐
    """
    
    def __init__(self, data_path="goodbooks-10k-master"):
        """
        初始化推荐系统
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.books_df = None
        self.ratings_df = None
        self.book_tags_df = None
        self.tags_df = None
        self.to_read_df = None
        
        # 推荐模型
        self.content_similarity_matrix = None
        self.collaborative_model = None
        
        # 特征工程
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.scaler = StandardScaler()
        
        # 用户交互记录
        self.user_interactions = {}
        
    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据...")
        
        try:
            # 加载主要数据文件
            self.books_df = pd.read_csv(f"{self.data_path}/books.csv")
            self.ratings_df = pd.read_csv(f"{self.data_path}/ratings.csv")
            self.book_tags_df = pd.read_csv(f"{self.data_path}/book_tags.csv")
            self.tags_df = pd.read_csv(f"{self.data_path}/tags.csv")
            self.to_read_df = pd.read_csv(f"{self.data_path}/to_read.csv")
            
            print(f"数据加载完成:")
            print(f"- 图书数量: {len(self.books_df)}")
            print(f"- 评分数量: {len(self.ratings_df)}")
            print(f"- 用户数量: {self.ratings_df['user_id'].nunique()}")
            print(f"- 标签数量: {len(self.tags_df)}")
            
        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            print("请确保数据文件在正确的位置")
            return False
            
        return True
    
    def preprocess_data(self):
        """数据预处理"""
        print("正在进行数据预处理...")
        
        # 处理图书数据
        self.books_df['authors'] = self.books_df['authors'].fillna('Unknown')
        self.books_df['title'] = self.books_df['title'].fillna('Unknown')
        self.books_df['language_code'] = self.books_df['language_code'].fillna('eng')
        
        # 处理评分数据
        self.ratings_df = self.ratings_df.dropna()
        
        # 合并标签信息
        self._merge_tags()
        
        # 创建图书特征
        self._create_book_features()
        
        print("数据预处理完成")
    
    def _merge_tags(self):
        """合并图书标签信息"""
        # 合并标签名称
        book_tags_with_names = self.book_tags_df.merge(
            self.tags_df, on='tag_id', how='left'
        )
        
        # 按图书分组，合并标签
        book_tags_grouped = book_tags_with_names.groupby('goodreads_book_id').agg({
            'tag_name': lambda x: ' '.join(x.astype(str)),
            'count': 'sum'
        }).reset_index()
        
        # 合并到图书数据
        self.books_df = self.books_df.merge(
            book_tags_grouped, 
            left_on='goodreads_book_id', 
            right_on='goodreads_book_id', 
            how='left'
        )
        
        # 填充缺失的标签
        self.books_df['tag_name'] = self.books_df['tag_name'].fillna('')
    
    def _create_book_features(self):
        """创建图书特征"""
        # 创建文本特征
        self.books_df['text_features'] = (
            self.books_df['title'] + ' ' + 
            self.books_df['authors'] + ' ' + 
            self.books_df['tag_name']
        )
        
        # 数值特征
        self.books_df['publication_year'] = pd.to_numeric(
            self.books_df['original_publication_year'], errors='coerce'
        ).fillna(2000)
        
        # 评分特征
        self.books_df['average_rating'] = self.books_df['average_rating'].fillna(3.0)
        self.books_df['ratings_count'] = self.books_df['ratings_count'].fillna(0)
        
        # 创建流行度特征
        self.books_df['popularity'] = (
            self.books_df['average_rating'] * np.log1p(self.books_df['ratings_count'])
        )
    
    def build_content_based_model(self):
        """构建基于内容的推荐模型"""
        print("正在构建基于内容的推荐模型...")
        
        # TF-IDF向量化
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # 减少特征数量以提高性能
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.books_df['text_features']
        )
        
        # 计算相似度矩阵
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print("基于内容的推荐模型构建完成")
    
    def build_collaborative_model(self):
        """构建协同过滤推荐模型"""
        print("正在构建协同过滤推荐模型...")
        
        # 创建用户-图书评分矩阵
        rating_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='book_id', 
            values='rating'
        ).fillna(0)
        
        # 使用SVD降维
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        user_features = self.svd_model.fit_transform(rating_matrix)
        
        # 训练随机森林模型预测评分
        X, y = self._prepare_collaborative_features()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        self.collaborative_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.collaborative_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.collaborative_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"协同过滤模型评估 - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    def _prepare_collaborative_features(self):
        """准备协同过滤模型的特征"""
        # 用户特征
        user_features = self.ratings_df.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std'],
            'book_id': 'nunique'
        }).reset_index()
        
        user_features.columns = ['user_id', 'user_avg_rating', 'user_rating_count', 
                               'user_rating_std', 'user_unique_books']
        
        # 图书特征
        book_features = self.books_df[['book_id', 'average_rating', 'ratings_count', 
                                     'popularity', 'publication_year']].copy()
        
        # 合并特征
        features = self.ratings_df.merge(user_features, on='user_id', how='left')
        features = features.merge(book_features, on='book_id', how='left')
        
        # 目标变量
        y = features['rating']
        X = features.drop(['user_id', 'book_id', 'rating'], axis=1)
        
        return X, y
    
    def get_content_recommendations(self, book_id, n_recommendations=10):
        """基于内容的推荐"""
        if self.content_similarity_matrix is None:
            print("请先构建基于内容的推荐模型")
            return []
        
        # 找到图书索引
        book_idx = self.books_df[self.books_df['book_id'] == book_id].index
        if len(book_idx) == 0:
            print(f"未找到图书ID: {book_id}")
            return []
        
        book_idx = book_idx[0]
        
        # 计算相似度
        sim_scores = list(enumerate(self.content_similarity_matrix[book_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # 获取推荐图书
        sim_scores = sim_scores[1:n_recommendations+1]
        book_indices = [i[0] for i in sim_scores]
        
        recommendations = self.books_df.iloc[book_indices][
            ['book_id', 'title', 'authors', 'average_rating']
        ].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """协同过滤推荐"""
        if self.collaborative_model is None:
            print("请先构建协同过滤推荐模型")
            return []
        
        # 获取用户未读过的图书
        user_rated_books = set(
            self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id']
        )
        all_books = set(self.books_df['book_id'])
        unrated_books = list(all_books - user_rated_books)
        
        # 准备预测特征
        predictions = []
        for book_id in unrated_books[:500]:  # 限制数量以提高速度
            pred_features = self._get_prediction_features(user_id, book_id)
            if pred_features is not None:
                pred_rating = self.collaborative_model.predict([pred_features])[0]
                predictions.append((book_id, pred_rating))
        
        # 排序并获取推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_book_ids = [pred[0] for pred in predictions[:n_recommendations]]
        
        recommendations = self.books_df[
            self.books_df['book_id'].isin(recommended_book_ids)
        ][['book_id', 'title', 'authors', 'average_rating']].copy()
        
        # 添加预测评分
        pred_scores = dict(predictions[:n_recommendations])
        recommendations['predicted_rating'] = recommendations['book_id'].map(pred_scores)
        
        return recommendations.sort_values('predicted_rating', ascending=False)
    
    def _get_prediction_features(self, user_id, book_id):
        """获取预测特征"""
        try:
            # 用户特征
            user_data = self.ratings_df[self.ratings_df['user_id'] == user_id]
            if len(user_data) == 0:
                return None
            
            user_avg_rating = user_data['rating'].mean()
            user_rating_count = len(user_data)
            user_rating_std = user_data['rating'].std()
            user_unique_books = user_data['book_id'].nunique()
            
            # 图书特征
            book_data = self.books_df[self.books_df['book_id'] == book_id]
            if len(book_data) == 0:
                return None
            
            book_avg_rating = book_data['average_rating'].iloc[0]
            book_ratings_count = book_data['ratings_count'].iloc[0]
            book_popularity = book_data['popularity'].iloc[0]
            book_publication_year = book_data['publication_year'].iloc[0]
            
            return [
                user_avg_rating, user_rating_count, user_rating_std, user_unique_books,
                book_avg_rating, book_ratings_count, book_popularity, book_publication_year
            ]
        except:
            return None
    
    def get_popular_recommendations(self, n_recommendations=10):
        """获取热门图书推荐"""
        popular_books = self.books_df.nlargest(n_recommendations, 'popularity')[
            ['book_id', 'title', 'authors', 'average_rating', 'ratings_count']
        ]
        return popular_books
    
    def get_personalized_recommendations(self, user_id, n_recommendations=10):
        """获取个性化推荐（混合方法）"""
        # 检查用户是否有交互历史
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            # 新用户，返回热门推荐
            print("新用户，返回热门图书推荐")
            return self.get_popular_recommendations(n_recommendations)
        
        # 有交互历史的用户，使用混合推荐
        recommendations = []
        
        # 协同过滤推荐
        cf_recs = self.get_collaborative_recommendations(user_id, n_recommendations//2)
        if len(cf_recs) > 0:
            cf_recs['recommendation_type'] = 'collaborative'
            recommendations.append(cf_recs)
        
        # 基于用户最喜欢的图书进行内容推荐
        user_favorite = user_ratings.loc[user_ratings['rating'].idxmax(), 'book_id']
        content_recs = self.get_content_recommendations(user_favorite, n_recommendations//2)
        if len(content_recs) > 0:
            content_recs['recommendation_type'] = 'content'
            recommendations.append(content_recs)
        
        if recommendations:
            final_recs = pd.concat(recommendations, ignore_index=True)
            final_recs = final_recs.drop_duplicates(subset=['book_id']).head(n_recommendations)
            return final_recs
        
        # 如果没有推荐，返回热门图书
        return self.get_popular_recommendations(n_recommendations)
    
    def record_user_interaction(self, user_id, book_id, interaction_type='view'):
        """记录用户交互"""
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        
        self.user_interactions[user_id].append({
            'book_id': book_id,
            'interaction_type': interaction_type,
            'timestamp': pd.Timestamp.now()
        })
    
    def save_models(self, model_path="models"):
        """保存模型"""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # 保存内容推荐模型
        if self.content_similarity_matrix is not None:
            joblib.dump(self.content_similarity_matrix, f"{model_path}/content_similarity.pkl")
            joblib.dump(self.tfidf_vectorizer, f"{model_path}/tfidf_vectorizer.pkl")
        
        # 保存协同过滤模型
        if self.collaborative_model is not None:
            joblib.dump(self.collaborative_model, f"{model_path}/collaborative_model.pkl")
            joblib.dump(self.svd_model, f"{model_path}/svd_model.pkl")
        
        # 保存数据
        joblib.dump(self.books_df, f"{model_path}/books_df.pkl")
        joblib.dump(self.ratings_df, f"{model_path}/ratings_df.pkl")
        
        print(f"模型已保存到 {model_path}")
    
    def load_models(self, model_path="models"):
        """加载模型"""
        try:
            # 加载内容推荐模型
            self.content_similarity_matrix = joblib.load(f"{model_path}/content_similarity.pkl")
            self.tfidf_vectorizer = joblib.load(f"{model_path}/tfidf_vectorizer.pkl")
            
            # 加载协同过滤模型
            self.collaborative_model = joblib.load(f"{model_path}/collaborative_model.pkl")
            self.svd_model = joblib.load(f"{model_path}/svd_model.pkl")
            
            # 加载数据
            self.books_df = joblib.load(f"{model_path}/books_df.pkl")
            self.ratings_df = joblib.load(f"{model_path}/ratings_df.pkl")
            
            print("模型加载成功")
            return True
            
        except FileNotFoundError:
            print("模型文件未找到，请先训练模型")
            return False
    
    def train_all_models(self):
        """训练所有模型"""
        print("开始训练所有推荐模型...")
        
        # 加载和预处理数据
        if not self.load_data():
            return False
        
        self.preprocess_data()
        
        # 训练各种模型
        self.build_content_based_model()
        self.build_collaborative_model()
        
        # 保存模型
        self.save_models()
        
        print("所有模型训练完成！")
        return True

