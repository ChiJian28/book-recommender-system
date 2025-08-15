#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图书推荐系统快速演示
"""

from book_recommender_simple import SimpleBookRecommender
import pandas as pd

def main():
    """主演示函数"""
    print("=" * 60)
    print("📚 图书推荐系统 - 快速演示")
    print("=" * 60)
    
    # 初始化推荐系统
    recommender = SimpleBookRecommender()
    
    # 加载数据
    print("\n📊 加载数据...")
    if not recommender.load_data():
        print("❌ 数据加载失败")
        return
    
    # 数据预处理
    print("\n🔧 数据预处理...")
    recommender.preprocess_data()
    
    # 显示数据统计
    print("\n📈 数据统计:")
    print(f"  - 图书总数: {len(recommender.books_df):,}")
    print(f"  - 用户总数: {recommender.ratings_df['user_id'].nunique():,}")
    print(f"  - 评分总数: {len(recommender.ratings_df):,}")
    print(f"  - 平均评分: {recommender.books_df['average_rating'].mean():.2f}")
    
    # 训练内容推荐模型
    print("\n🎯 训练基于内容的推荐模型...")
    recommender.build_content_based_model()
    
    # 演示热门推荐
    print("\n🔥 热门图书推荐 (Top 5):")
    popular_books = recommender.get_popular_recommendations(5)
    for i, (_, book) in enumerate(popular_books.iterrows(), 1):
        print(f"  {i}. {book['title']}")
        print(f"     作者: {book['authors']}")
        print(f"     评分: {book['average_rating']:.2f} ({book['ratings_count']:,} 人评分)")
        print()
    
    # 演示基于内容的推荐
    print("\n🔍 基于内容的推荐演示:")
    # 选择一个热门图书进行内容推荐
    test_book = popular_books.iloc[0]
    print(f"基于图书: {test_book['title']}")
    
    content_recs = recommender.get_content_recommendations(test_book['book_id'], 5)
    for i, (_, book) in enumerate(content_recs.iterrows(), 1):
        print(f"  {i}. {book['title']}")
        print(f"     作者: {book['authors']}")
        print(f"     相似度: {book['similarity_score']:.3f}")
        print()
    
    # 演示新用户推荐
    print("\n🆕 新用户推荐演示:")
    new_user_id = 999999  # 不存在的用户ID
    new_user_recs = recommender.get_personalized_recommendations(new_user_id, 5)
    print(f"新用户 (ID: {new_user_id}) 的推荐:")
    for i, (_, book) in enumerate(new_user_recs.iterrows(), 1):
        print(f"  {i}. {book['title']}")
        print(f"     作者: {book['authors']}")
        print(f"     评分: {book['average_rating']:.2f}")
        print()
    
    # 显示一些有趣的统计
    print("\n📊 有趣的统计信息:")
    
    # 最受欢迎的图书
    most_popular = recommender.books_df.nlargest(1, 'ratings_count')
    print(f"  - 最受欢迎的图书: {most_popular['title'].iloc[0]}")
    print(f"    评分人数: {most_popular['ratings_count'].iloc[0]:,}")
    
    # 最高评分的图书
    highest_rated = recommender.books_df.nlargest(1, 'average_rating')
    print(f"  - 最高评分的图书: {highest_rated['title'].iloc[0]}")
    print(f"    平均评分: {highest_rated['average_rating'].iloc[0]:.2f}")
    
    # 最活跃的用户
    most_active_user = recommender.ratings_df.groupby('user_id').size().idxmax()
    user_rating_count = recommender.ratings_df.groupby('user_id').size().max()
    print(f"  - 最活跃的用户: ID {most_active_user}")
    print(f"    评分数量: {user_rating_count:,}")
    
    print("\n🎉 演示完成！")
    print("\n💡 提示:")
    print("  - 运行 'python app.py' 启动Web界面")
    print("  - 运行 'python demo.py' 进行交互式演示")
    print("  - 运行 'python train_models.py' 训练完整模型")

if __name__ == "__main__":
    main()

