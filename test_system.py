#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图书推荐系统测试脚本
"""

import sys
import os
from book_recommender_simple import SimpleBookRecommender as BookRecommender

def test_data_loading():
    """测试数据加载功能"""
    print("🧪 测试数据加载...")
    
    recommender = BookRecommender()
    
    # 测试数据加载
    success = recommender.load_data()
    if not success:
        print("❌ 数据加载失败")
        return False
    
    print("✅ 数据加载成功")
    print(f"  - 图书数量: {len(recommender.books_df)}")
    print(f"  - 评分数量: {len(recommender.ratings_df)}")
    print(f"  - 用户数量: {recommender.ratings_df['user_id'].nunique()}")
    
    return True

def test_data_preprocessing():
    """测试数据预处理功能"""
    print("\n🧪 测试数据预处理...")
    
    recommender = BookRecommender()
    recommender.load_data()
    
    # 测试数据预处理
    try:
        recommender.preprocess_data()
        print("✅ 数据预处理成功")
        
        # 检查预处理后的数据
        print(f"  - 图书特征数量: {len(recommender.books_df.columns)}")
        print(f"  - 文本特征示例: {recommender.books_df['text_features'].iloc[0][:100]}...")
        print(f"  - 流行度范围: {recommender.books_df['popularity'].min():.2f} - {recommender.books_df['popularity'].max():.2f}")
        
        return True
    except Exception as e:
        print(f"❌ 数据预处理失败: {str(e)}")
        return False

def test_model_training():
    """测试模型训练功能"""
    print("\n🧪 测试模型训练...")
    
    recommender = BookRecommender()
    recommender.load_data()
    recommender.preprocess_data()
    
    try:
        # 测试内容推荐模型
        print("  训练基于内容的推荐模型...")
        recommender.build_content_based_model()
        print("  ✅ 内容推荐模型训练成功")
        
        # 测试协同过滤模型（使用小数据集）
        print("  训练协同过滤模型...")
        # 为了快速测试，使用部分数据
        sample_ratings = recommender.ratings_df.head(10000)  # 使用前1万条评分
        original_ratings = recommender.ratings_df
        recommender.ratings_df = sample_ratings
        
        recommender.build_collaborative_model()
        print("  ✅ 协同过滤模型训练成功")
        
        # 恢复原始数据
        recommender.ratings_df = original_ratings
        
        return True
    except Exception as e:
        print(f"❌ 模型训练失败: {str(e)}")
        return False

def test_recommendations():
    """测试推荐功能"""
    print("\n🧪 测试推荐功能...")
    
    recommender = BookRecommender()
    recommender.load_data()
    recommender.preprocess_data()
    
    try:
        # 测试热门推荐
        print("  测试热门图书推荐...")
        popular_books = recommender.get_popular_recommendations(5)
        if len(popular_books) > 0:
            print(f"  ✅ 热门推荐成功，返回 {len(popular_books)} 本图书")
        else:
            print("  ❌ 热门推荐失败")
            return False
        
        # 测试内容推荐
        print("  测试基于内容的推荐...")
        recommender.build_content_based_model()
        test_book_id = recommender.books_df.iloc[0]['book_id']
        content_recs = recommender.get_content_recommendations(test_book_id, 3)
        if len(content_recs) > 0:
            print(f"  ✅ 内容推荐成功，返回 {len(content_recs)} 本相似图书")
        else:
            print("  ❌ 内容推荐失败")
            return False
        
        # 测试个性化推荐
        print("  测试个性化推荐...")
        if len(recommender.ratings_df) > 0:
            test_user_id = recommender.ratings_df.iloc[0]['user_id']
            personalized_recs = recommender.get_personalized_recommendations(test_user_id, 3)
            if len(personalized_recs) > 0:
                print(f"  ✅ 个性化推荐成功，返回 {len(personalized_recs)} 本推荐图书")
            else:
                print("  ❌ 个性化推荐失败")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 推荐功能测试失败: {str(e)}")
        return False

def test_model_saving():
    """测试模型保存功能"""
    print("\n🧪 测试模型保存...")
    
    recommender = BookRecommender()
    recommender.load_data()
    recommender.preprocess_data()
    recommender.build_content_based_model()
    
    try:
        # 测试模型保存
        recommender.save_models("test_models")
        print("✅ 模型保存成功")
        
        # 测试模型加载
        new_recommender = BookRecommender()
        success = new_recommender.load_models("test_models")
        if success:
            print("✅ 模型加载成功")
            
            # 清理测试文件
            import shutil
            if os.path.exists("test_models"):
                shutil.rmtree("test_models")
            print("✅ 测试文件清理完成")
            
            return True
        else:
            print("❌ 模型加载失败")
            return False
    except Exception as e:
        print(f"❌ 模型保存/加载失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 图书推荐系统测试")
    print("=" * 60)
    
    tests = [
        ("数据加载", test_data_loading),
        ("数据预处理", test_data_preprocessing),
        ("模型训练", test_model_training),
        ("推荐功能", test_recommendations),
        ("模型保存", test_model_saving)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
