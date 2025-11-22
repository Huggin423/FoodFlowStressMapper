# 外卖骑手压力区域识别与预测项目

## 项目简介

本项目是"智慧城市"课程作业，旨在通过分析外卖骑手的配送路线数据，识别和预测骑手压力区域。项目采用数据驱动的方法，结合机器学习模型和时空分析技术，为外卖平台提供骑手压力预警和调度优化建议。

## 快速开始

### 环境要求

- Python 3.8+
- 见 `requirements.txt` 中的依赖包

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

#### 1. 数据预处理（如果数据更新）

```bash
python process/data_preprocessor.py
```

#### 2. 压力分数计算（如果预处理数据更新）

```bash
python stress_scoring/智慧城市2(1).py
```

#### 3. 模型训练

```bash
python prediction/train.py
```

#### 4. 热点聚类分析

```bash
python prediction/hotspot_cluster.py
```

#### 5. SHAP可解释性分析（需要先训练模型）

```bash
python prediction/shap_explain.py
```

## 项目结构

```
rider_stress_prediction/
├── data/                      # 数据目录
├── process/                   # 数据预处理模块
├── stress_scoring/            # 压力分数计算模块
├── prediction/                # 预测模型模块
├── prediction_output/         # 模型输出
├── plan/                      # 项目计划文档
└── README.md                  # 本文档
```

详细说明请参阅：**[项目完整说明文档.md](项目完整说明文档.md)**

## 核心功能

1. ✅ **多维度特征提取**：行为、时序、空间、环境四类特征
2. ✅ **层级联动压力计算**：Route和Action_Point两级压力建模
3. ✅ **机器学习优化**：XGBoost自动学习特征权重
4. ✅ **时空预测模型**：XGBoost基线和STGCN深度学习模型
5. ✅ **热点识别**：ST-DBSCAN识别时空热点区域
6. ✅ **可解释性分析**：SHAP提供模型解释

## 输出文件

### 压力分数计算输出
- `rider_stress_hierarchical_final.csv` - 层级压力分数
- `xgb_optimized_stress.csv` - XGBoost优化压力分数
- `xgb_optimized_weights.csv` - 特征权重
- `xgb_feature_importance.png` - 特征重要性图

### 模型训练输出
- `prediction_output/xgb_model.json` - XGBoost模型
- `prediction_output/training_report.json` - 训练报告
- `prediction_output/stgcn.pt` - STGCN模型（如果使用）

### 热点聚类输出
- `prediction_output/hotspot_points.csv` - 热点聚类点
- `prediction_output/hotspot.geojson` - 热点GeoJSON
- `prediction_output/hotspot_cluster_summary.csv` - 聚类统计

### SHAP分析输出
- `prediction_output/shap_values.npy` - SHAP值
- `prediction_output/shap_feature_importance.csv` - SHAP特征重要性
- `prediction_output/shap_summary.png` - SHAP可视化

## 项目状态

✅ **已完成**
- 数据预处理模块
- 压力分数计算模块
- 预测模型框架
- 热点聚类实现
- SHAP可解释性分析

⚠️ **待优化**
- 模型超参数调优
- 热点聚类质量验证
- 可视化增强

## 技术栈

- **数据处理**：pandas, numpy
- **机器学习**：scikit-learn, xgboost
- **深度学习**：PyTorch (可选，用于STGCN)
- **可解释性**：SHAP
- **空间分析**：geopandas
- **可视化**：matplotlib

## 详细文档

完整的使用说明、数据流程、代码解释等，请参阅：

- **[项目完整说明文档.md](项目完整说明文档.md)** - 详细的项目说明
- **plan/** 目录 - 项目计划文档

## 许可证

本项目为课程作业项目。

## 联系方式

如有问题，请联系项目组。

---

**最后更新：** 2024年

