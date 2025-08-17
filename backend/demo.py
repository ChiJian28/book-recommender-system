#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图书推荐系统演示脚本
"""

from models.book_recommender_simple import SimpleBookRecommender as BookRecommender
import pandas as pd

def main():
    """主演示函数"""
    print("=" * 60)
    print("📚 图书推荐系统演示")
    print("=" * 60)
    
    # 初始化推荐系统
    recommender = BookRecommender()
    
    # 尝试加载已训练的模型
    if not recommender.load_models():
        print("❌ 未找到已训练的模型")
        print("请先运行 'python train_models.py' 训练模型")
        return
    
    print("✅ 模型加载成功！")
    
    # 显示系统信息
    show_system_info(recommender)
    
    # 演示各种推荐功能
    demonstrate_recommendations(recommender)

def show_system_info(recommender):
    """显示系统信息"""
    print("\n📊 系统信息:")
    print(f"  - 图书总数: {len(recommender.books_df):,}")
    print(f"  - 用户总数: {recommender.ratings_df['user_id'].nunique():,}")
    print(f"  - 评分总数: {len(recommender.ratings_df):,}")
    print(f"  - 标签总数: {len(recommender.tags_df):,}")
    
    # 显示一些统计信息
    print(f"\n📈 数据统计:")
    print(f"  - 平均评分: {recommender.books_df['average_rating'].mean():.2f}")
    print(f"  - 最高评分: {recommender.books_df['average_rating'].max():.2f}")
    print(f"  - 最低评分: {recommender.books_df['average_rating'].min():.2f}")
    print(f"  - 平均出版年份: {recommender.books_df['publication_year'].mean():.0f}")

def demonstrate_recommendations(recommender):
    """演示各种推荐功能"""
    print("\n" + "=" * 60)
    print("🎯 推荐功能演示")
    print("=" * 60)
    
    # 1. 热门图书推荐
    print("\n🔥 热门图书推荐 (Top 5):")
    popular_books = recommender.get_popular_recommendations(5)
    for i, (_, book) in enumerate(popular_books.iterrows(), 1):
        print(f"  {i}. {book['title']}")
        print(f"     作者: {book['authors']}")
        print(f"     评分: {book['average_rating']:.2f} ({book['ratings_count']:,} 人评分)")
        print()
    
    # 2. 基于内容的推荐
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
    
    # 3. 个性化推荐
    print("\n👤 个性化推荐演示:")
    # 选择一个有较多评分的用户
    active_users = recommender.ratings_df.groupby('user_id').size().sort_values(ascending=False)
    if len(active_users) > 0:
        test_user_id = active_users.index[0]
        user_rating_count = active_users.iloc[0]
        print(f"用户ID: {test_user_id} (有 {user_rating_count} 条评分记录)")
        
        personalized_recs = recommender.get_personalized_recommendations(test_user_id, 5)
        for i, (_, book) in enumerate(personalized_recs.iterrows(), 1):
            print(f"  {i}. {book['title']}")
            print(f"     作者: {book['authors']}")
            print(f"     评分: {book['average_rating']:.2f}")
            if 'predicted_rating' in book:
                print(f"     预测评分: {book['predicted_rating']:.2f}")
            print()
    
    # 4. 新用户推荐
    print("\n🆕 新用户推荐演示:")
    # 使用一个不存在的用户ID
    new_user_id = 999999
    new_user_recs = recommender.get_personalized_recommendations(new_user_id, 5)
    print(f"新用户 (ID: {new_user_id}) 的推荐:")
    for i, (_, book) in enumerate(new_user_recs.iterrows(), 1):
        print(f"  {i}. {book['title']}")
        print(f"     作者: {book['authors']}")
        print(f"     评分: {book['average_rating']:.2f}")
        print()

def show_book_details(recommender, book_id):
    """显示图书详细信息"""
    book = recommender.books_df[recommender.books_df['book_id'] == book_id]
    if len(book) == 0:
        print(f"未找到图书ID: {book_id}")
        return
    
    book = book.iloc[0]
    print(f"\n📖 图书详情:")
    print(f"  标题: {book['title']}")
    print(f"  作者: {book['authors']}")
    print(f"  出版年份: {book['publication_year']}")
    print(f"  平均评分: {book['average_rating']:.2f}")
    print(f"  评分人数: {book['ratings_count']:,}")
    print(f"  语言: {book['language_code']}")
    print(f"  流行度: {book['popularity']:.2f}")

def interactive_demo(recommender):
    """交互式演示"""
    print("\n" + "=" * 60)
    print("🎮 交互式演示")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("1. 查看图书详情")
        print("2. 基于图书推荐")
        print("3. 用户个性化推荐")
        print("4. 查看热门图书")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            book_id = input("请输入图书ID: ").strip()
            try:
                show_book_details(recommender, int(book_id))
            except ValueError:
                print("请输入有效的图书ID")
        
        elif choice == '2':
            book_id = input("请输入图书ID: ").strip()
            try:
                content_recs = recommender.get_content_recommendations(int(book_id), 5)
                print(f"\n基于图书ID {book_id} 的推荐:")
                for i, (_, book) in enumerate(content_recs.iterrows(), 1):
                    print(f"  {i}. {book['title']} (相似度: {book['similarity_score']:.3f})")
            except ValueError:
                print("请输入有效的图书ID")
        
        elif choice == '3':
            user_id = input("请输入用户ID: ").strip()
            try:
                personalized_recs = recommender.get_personalized_recommendations(int(user_id), 5)
                print(f"\n用户 {user_id} 的个性化推荐:")
                for i, (_, book) in enumerate(personalized_recs.iterrows(), 1):
                    print(f"  {i}. {book['title']}")
            except ValueError:
                print("请输入有效的用户ID")
        
        elif choice == '4':
            popular_books = recommender.get_popular_recommendations(10)
            print(f"\n热门图书 Top 10:")
            for i, (_, book) in enumerate(popular_books.iterrows(), 1):
                print(f"  {i}. {book['title']} (评分: {book['average_rating']:.2f})")
        
        elif choice == '5':
            print("感谢使用图书推荐系统！")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
    
    # 询问是否进行交互式演示
    response = input("\n是否进行交互式演示？(y/n): ").strip().lower()
    if response == 'y':
        recommender = BookRecommender()
        if recommender.load_models():
            interactive_demo(recommender)
        else:
            print("无法加载模型，跳过交互式演示")
