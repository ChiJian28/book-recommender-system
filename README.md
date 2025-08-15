# 📚 个性化图书推荐系统

一个基于机器学习的个性化图书推荐系统，支持多种推荐算法，包括协同过滤、内容推荐和混合推荐。

## 🎯 功能特性

- **多种推荐算法**：
  - 基于内容的推荐（Content-Based）
  - 协同过滤推荐（Collaborative Filtering）
  - 混合推荐（Hybrid）
  - 热门图书推荐

- **智能推荐策略**：
  - 新用户冷启动处理
  - 个性化推荐
  - 实时用户交互记录

- **Web界面**：
  - 友好的用户界面
  - RESTful API接口
  - 实时推荐结果展示

## 📋 系统要求

- Python 3.10+
- 依赖包（见 `requirement.txt`）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirement.txt
```

### 2. 训练模型

```bash
python train_models.py
```

这将：
- 加载和预处理数据
- 训练基于内容的推荐模型
- 训练协同过滤模型
- 训练混合推荐模型
- 保存所有模型到 `models/` 目录

### 3. 启动Web服务

```bash
python app.py
```

然后在浏览器中访问：http://localhost:5000

### 4. 演示系统功能

```bash
python demo.py
```

## 📁 项目结构

```
Recommender System/
├── book_recommender.py      # 核心推荐系统类
├── app.py                   # Flask Web应用
├── train_models.py          # 模型训练脚本
├── demo.py                  # 演示脚本
├── requirement.txt          # 依赖包列表
├── README.md               # 项目说明
├── goodbooks-10k-master/   # 数据集目录
│   ├── books.csv           # 图书信息
│   ├── ratings.csv         # 用户评分
│   ├── book_tags.csv       # 图书标签
│   ├── tags.csv            # 标签信息
│   └── to_read.csv         # 用户想读列表
└── models/                 # 训练好的模型（自动创建）
```

## 🔧 核心组件

### BookRecommender 类

主要的推荐系统类，包含以下方法：

- `load_data()`: 加载数据文件
- `preprocess_data()`: 数据预处理
- `build_content_based_model()`: 构建基于内容的推荐模型
- `build_collaborative_model()`: 构建协同过滤模型
- `build_hybrid_model()`: 构建混合推荐模型
- `get_content_recommendations()`: 获取基于内容的推荐
- `get_collaborative_recommendations()`: 获取协同过滤推荐
- `get_personalized_recommendations()`: 获取个性化推荐
- `get_popular_recommendations()`: 获取热门图书推荐

### Web API 接口

- `GET /`: 主页面
- `GET /api/content_recommendations/<book_id>`: 基于内容的推荐
- `GET /api/personalized_recommendations/<user_id>`: 个性化推荐
- `GET /api/popular_books`: 热门图书推荐
- `GET /api/system_info`: 系统信息
- `GET /api/search_books`: 搜索图书
- `POST /api/user_interaction`: 记录用户交互

## 🎮 使用示例

### 基本使用

```python
from book_recommender import BookRecommender

# 初始化推荐系统
recommender = BookRecommender()

# 加载已训练的模型
recommender.load_models()

# 获取热门图书推荐
popular_books = recommender.get_popular_recommendations(10)

# 获取基于内容的推荐
content_recs = recommender.get_content_recommendations(book_id=1, n_recommendations=5)

# 获取个性化推荐
personalized_recs = recommender.get_personalized_recommendations(user_id=1, n_recommendations=10)
```

### Web界面使用

1. 启动Web服务后，访问 http://localhost:5000
2. 在"基于图书的推荐"部分输入图书ID，获取相似图书
3. 在"个性化推荐"部分输入用户ID，获取个性化推荐
4. 点击"获取热门图书"查看当前最热门的图书

## 📊 数据集说明

使用 Goodreads 10K 数据集，包含：

- **books.csv**: 10,000本图书的基本信息
- **ratings.csv**: 用户评分数据
- **book_tags.csv**: 图书标签关联
- **tags.csv**: 标签信息
- **to_read.csv**: 用户想读列表

## 🔍 推荐算法详解

### 1. 基于内容的推荐（Content-Based）

- 使用TF-IDF向量化图书的标题、作者和标签
- 计算图书间的余弦相似度
- 基于相似度推荐相似图书

### 2. 协同过滤推荐（Collaborative Filtering）

- 使用SVD（奇异值分解）算法
- 基于用户-图书评分矩阵
- 预测用户对未评分图书的评分

### 3. 混合推荐（Hybrid）

- 结合用户特征和图书特征
- 使用LightGBM梯度提升模型
- 融合多种推荐策略

### 4. 热门推荐

- 基于图书的流行度评分
- 考虑平均评分和评分人数
- 适合新用户冷启动

## ⚙️ 配置选项

可以在 `book_recommender.py` 中调整以下参数：

- TF-IDF特征数量：`max_features=5000`
- SVD隐因子数量：`n_factors=100`
- LightGBM参数：`n_estimators=100`, `learning_rate=0.1`
- 推荐数量：默认10本

## 🐛 故障排除

### 常见问题

1. **模型文件未找到**
   - 确保已运行 `python train_models.py`
   - 检查 `models/` 目录是否存在

2. **内存不足**
   - 减少TF-IDF的 `max_features` 参数
   - 使用数据集的子集进行测试

3. **训练时间过长**
   - 可以使用样本数据进行快速测试
   - 调整模型参数减少训练时间

### 性能优化

- 使用更小的数据集进行开发和测试
- 调整模型参数平衡性能和准确性
- 考虑使用更高效的算法实现

## 📈 模型评估

系统会自动评估模型性能：

- **协同过滤模型**: RMSE和MAE指标
- **混合推荐模型**: RMSE和MAE指标
- **内容推荐**: 基于相似度矩阵

## 🔄 模型更新

系统支持模型的重训练和更新：

```python
# 重新训练所有模型
recommender.train_all_models()

# 保存模型
recommender.save_models()
```

## 📝 开发计划

- [ ] 添加更多推荐算法（如深度学习模型）
- [ ] 实现实时推荐更新
- [ ] 添加推荐解释功能
- [ ] 支持更多数据源
- [ ] 优化Web界面
- [ ] 添加用户认证系统

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

- Goodreads数据集提供者
- 开源机器学习库的贡献者
- 推荐系统研究社区

