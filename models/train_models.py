#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图书推荐系统模型训练脚本
"""

from book_recommender_simple import SimpleBookRecommender as BookRecommender
import time
import sys

def main():
    """主函数"""
    print("=" * 50)
    print("📚 图书推荐系统 - 模型训练")
    print("=" * 50)
    
    # 初始化推荐系统
    recommender = BookRecommender()
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 训练所有模型
        print("\n🚀 开始训练推荐模型...")
        success = recommender.train_all_models()
        
        if success:
            # 计算训练时间
            training_time = time.time() - start_time
            print(f"\n✅ 模型训练完成！")
            print(f"⏱️  训练耗时: {training_time:.2f} 秒")
            
            # 测试推荐功能
            print("\n🧪 测试推荐功能...")
            test_recommendations(recommender)
            
            print("\n🎉 系统准备就绪！")
            print("运行 'python app.py' 启动Web服务")
            
        else:
            print("\n❌ 模型训练失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {str(e)}")
        sys.exit(1)

def test_recommendations(recommender):
    """测试推荐功能"""
    try:
        # 测试热门推荐
        print("测试热门图书推荐...")
        popular_books = recommender.get_popular_recommendations(5)
        if len(popular_books) > 0:
            print(f"✅ 热门推荐测试成功，返回 {len(popular_books)} 本图书")
            print("热门图书示例:")
            for i, book in popular_books.head(3).iterrows():
                print(f"  - {book['title']} (评分: {book['average_rating']})")
        else:
            print("❌ 热门推荐测试失败")
        
        # 测试内容推荐
        print("\n测试基于内容的推荐...")
        if len(recommender.books_df) > 0:
            test_book_id = recommender.books_df.iloc[0]['book_id']
            content_recs = recommender.get_content_recommendations(test_book_id, 3)
            if len(content_recs) > 0:
                print(f"✅ 内容推荐测试成功，返回 {len(content_recs)} 本相似图书")
                print("内容推荐示例:")
                for i, book in content_recs.head(3).iterrows():
                    print(f"  - {book['title']} (相似度: {book['similarity_score']:.3f})")
            else:
                print("❌ 内容推荐测试失败")
        
        # 测试个性化推荐
        print("\n测试个性化推荐...")
        if len(recommender.ratings_df) > 0:
            test_user_id = recommender.ratings_df.iloc[0]['user_id']
            personalized_recs = recommender.get_personalized_recommendations(test_user_id, 3)
            if len(personalized_recs) > 0:
                print(f"✅ 个性化推荐测试成功，返回 {len(personalized_recs)} 本推荐图书")
                print("个性化推荐示例:")
                for i, book in personalized_recs.head(3).iterrows():
                    print(f"  - {book['title']} (评分: {book['average_rating']})")
            else:
                print("❌ 个性化推荐测试失败")
        
        print("\n✅ 所有推荐功能测试完成！")
        
    except Exception as e:
        print(f"❌ 推荐功能测试失败: {str(e)}")

if __name__ == "__main__":
    main()
