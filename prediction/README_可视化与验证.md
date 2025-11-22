# 可视化与模型验证使用说明

## 📊 功能说明

### 1. 压力区域热力图可视化 (`visualize_heatmap.py`)

**功能：**
- 从原始GeoJSON文件提取路线坐标（经纬度）
- 合并XGBoost优化的压力分数
- 生成28天×5个时段的压力区域热力图
- 创建时间序列热力图（28天×5时段的压力变化）

**使用方法：**
```bash
# 在conda环境中运行
conda activate rider_stress
python prediction/visualize_heatmap.py
```

**输出文件：**
1. `prediction_output/heatmaps/` 目录：
   - 125张热力图（25天 × 5个时段）
   - 文件名格式：`heatmap_YYYY-MM-DD_时间段.png`
   - 每个图显示该日期该时段的压力分布（颜色=压力分数，大小=点数）

2. `prediction_output/stress_timeseries_heatmap.png`：
   - 28天×5时段的时间序列热力图
   - 横轴：日期，纵轴：时间段（早高峰、午高峰、白天平峰、晚高峰、夜间）
   - 颜色深浅表示平均压力分数

3. `prediction_output/heatmap_summary.csv`：
   - 每个日期×时段的压力统计（均值、标准差、最小值、最大值、样本数）

**时间段说明：**
- `morning_peak` - 早高峰（6-10点）
- `lunch_peak` - 午高峰（11-13点）
- `day_off_peak` - 白天平峰（10-17点）
- `dinner_peak` - 晚高峰（17-19点）
- `night` - 夜间（其他时间）

---

### 2. 模型验证 (`validate_model.py`)

**功能：**
- 时间序列分割（70%训练，30%测试）
- 预测值 vs 真实值对比分析
- 回归指标（MAE、RMSE、R²）
- 压力等级分类准确率
- 按时间段分析预测性能
- 生成多种可视化图表

**使用方法：**
```bash
# 在conda环境中运行（需要XGBoost）
conda activate rider_stress
python prediction/validate_model.py
```

**输出文件：**
1. `prediction_output/model_validation_predictions.png`：
   - 4个子图：
     - 预测值 vs 真实值散点图（含完美预测线）
     - 残差分析图
     - 时间序列对比（前100个样本）
     - 误差分布直方图

2. `prediction_output/model_validation_by_period.png`：
   - 4个子图：
     - 不同时间段的MAE
     - 不同时间段的平均压力分数（真实vs预测）
     - 不同时间段的误差分布（箱线图）
     - 不同时间段的R²分数

3. `prediction_output/model_validation_report.json`：
   - 详细的验证指标报告

**验证指标：**
- **回归指标：**
  - MAE（平均绝对误差）
  - RMSE（均方根误差）
  - R²（决定系数，越接近1越好）

- **分类指标：**
  - 压力等级分类准确率（低/中/高压力）

- **误差统计：**
  - 平均误差
  - 误差标准差
  - 最大高估/低估
  - MAPE（平均绝对百分比误差）

**模型效果判断：**
- ✅ R² > 0.8 且 MAE < 1.0：模型效果很好
- ⚠️ R² > 0.5 且 MAE < 2.0：模型有一定效果，可改进
- ❌ R² < 0.5 或 MAE > 2.0：模型效果较差，需要改进

---

## 📁 文件结构

```
prediction_output/
├── heatmaps/                          # 热力图目录（125张图）
│   ├── heatmap_2020-02-01_morning_peak.png
│   ├── heatmap_2020-02-01_lunch_peak.png
│   └── ...
├── stress_timeseries_heatmap.png      # 时间序列热力图
├── heatmap_summary.csv                # 热力图汇总统计
├── model_validation_predictions.png   # 预测验证图（4子图）
├── model_validation_by_period.png     # 时间段分析图（4子图）
└── model_validation_report.json       # 验证报告
```

---

## 🔍 结果解读

### 热力图解读

1. **单个热力图（heatmaps/）**：
   - **颜色**：从黄色到红色，表示压力分数从低到高（0-12分）
   - **点的大小**：表示该网格区域内路线数量（越大=路线越多）
   - **分布模式**：
     - 红色密集区域 = 高压区域
     - 黄色稀疏区域 = 低压区域

2. **时间序列热力图**：
   - **横轴**：日期（02-01 到 02-28）
   - **纵轴**：时间段（早高峰、午高峰、白天平峰、晚高峰、夜间）
   - **颜色深浅**：平均压力分数
   - **可以看出**：
     - 哪些时段压力较高（深红色）
     - 哪些日期压力较高（整列深色）
     - 压力随时间的变化趋势

### 模型验证解读

1. **预测值 vs 真实值散点图**：
   - 点越接近红色虚线（完美预测线）= 预测越准确
   - 点偏离红色虚线 = 预测有误差

2. **残差分析**：
   - 残差应该在0附近随机分布
   - 如果残差有趋势 = 模型存在系统性偏差

3. **时间段分析**：
   - 可以看出哪个时段预测效果好（MAE小、R²高）
   - 可以看出哪个时段预测困难（MAE大、R²低）

---

## ⚠️ 注意事项

1. **运行环境**：
   - 必须在conda环境 `rider_stress` 中运行
   - 需要安装XGBoost（用于模型验证）
   - 需要安装matplotlib（用于绘图）

2. **数据要求**：
   - 确保 `xgb_optimized_stress.csv` 文件存在（压力分数数据）
   - 确保 `data/raw/ODIDMob_Routes/` 目录中有GeoJSON文件（坐标数据）

3. **运行顺序**：
   - 先运行 `visualize_heatmap.py` 生成热力图
   - 再运行 `validate_model.py` 验证模型（需要先训练好模型）

4. **性能提示**：
   - 热力图生成需要一些时间（处理28天×5时段数据）
   - 如果数据量大，可能需要几分钟

---

## 📊 当前状态

✅ **已完成：**
- 热力图可视化脚本已创建并测试通过
- 生成了125张热力图
- 生成了时间序列热力图
- 模型验证脚本已创建

⚠️ **待运行：**
- 模型验证脚本需要在conda环境中运行（当前环境未检测到XGBoost）

---

## 🚀 快速开始

```bash
# 1. 激活conda环境
conda activate rider_stress

# 2. 生成热力图
python prediction/visualize_heatmap.py

# 3. 验证模型（如果模型已训练）
python prediction/validate_model.py

# 4. 查看结果
# - 热力图：prediction_output/heatmaps/
# - 时间序列图：prediction_output/stress_timeseries_heatmap.png
# - 验证图：prediction_output/model_validation_*.png
```

---

**最后更新：** 2024年

