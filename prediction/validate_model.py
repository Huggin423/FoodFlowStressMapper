"""验证预测模型是否起作用

通过以下方式验证模型：
1. 时间序列分割（train/test split）
2. 预测 vs 真实值对比
3. 按时间段分析预测性能
4. 可视化预测结果
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False

from data_loader import prepare_dataset
from evaluate import regression_metrics, classification_metrics, classify_level

OUTPUT_DIR = Path("prediction_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "xgb_model.json"


def time_series_split(X: np.ndarray, y: np.ndarray, meta: list, 
                     train_ratio: float = 0.7) -> tuple:
    """时间序列分割：按日期排序后分割训练集和测试集"""
    # 提取日期
    dates = [m['target_date'] for m in meta]
    
    # 按日期排序
    sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
    
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    meta_sorted = [meta[i] for i in sorted_indices]
    
    # 分割点
    split_idx = int(len(X_sorted) * train_ratio)
    
    X_train = X_sorted[:split_idx]
    y_train = y_sorted[:split_idx]
    meta_train = meta_sorted[:split_idx]
    
    X_test = X_sorted[split_idx:]
    y_test = y_sorted[split_idx:]
    meta_test = meta_sorted[split_idx:]
    
    return (X_train, y_train, meta_train), (X_test, y_test, meta_test)


def load_model():
    """加载训练好的模型"""
    if xgb is None:
        raise RuntimeError("XGBoost未安装")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def validate_predictions(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """验证预测结果"""
    y_pred = model.predict(X_test)
    
    # 回归指标
    reg_metrics = regression_metrics(y_test, y_pred)
    
    # 分类指标（压力等级）
    cls_metrics = classification_metrics(y_test, y_pred)
    
    # 误差统计
    errors = y_pred - y_test
    error_stats = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'max_overestimate': float(np.max(errors)),
        'max_underestimate': float(np.min(errors)),
        'mape': float(np.mean(np.abs(errors / (y_test + 1e-10))) * 100)  # MAPE
    }
    
    return {
        'regression': reg_metrics,
        'classification': cls_metrics,
        'error_stats': error_stats,
        'y_true': y_test,
        'y_pred': y_pred
    }


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: Path):
    """绘制预测值 vs 真实值"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 散点图
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    ax.set_xlabel('真实值', fontsize=11)
    ax.set_ylabel('预测值', fontsize=11)
    ax.set_title('预测值 vs 真实值（散点图）', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('预测值', fontsize=11)
    ax.set_ylabel('残差 (预测值 - 真实值)', fontsize=11)
    ax.set_title('残差分析', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 时间序列对比（如果有时间信息，这里用索引代替）
    ax = axes[1, 0]
    indices = np.arange(len(y_true))
    ax.plot(indices[:100], y_true[:100], 'o-', label='真实值', alpha=0.7, markersize=3)
    ax.plot(indices[:100], y_pred[:100], 's-', label='预测值', alpha=0.7, markersize=3)
    ax.set_xlabel('样本索引', fontsize=11)
    ax.set_ylabel('压力分数', fontsize=11)
    ax.set_title('预测值 vs 真实值（时间序列，前100个样本）', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 误差分布直方图
    ax = axes[1, 1]
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('残差', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title('误差分布直方图', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 保存预测对比图: {save_path}")


def plot_performance_by_time_period(all_data: pd.DataFrame, results: dict,
                                   save_path: Path):
    """按时间段分析预测性能"""
    # 合并预测结果到原始数据
    test_data = all_data[all_data['split'] == 'test'].copy()
    test_data['y_true'] = results['y_true']
    test_data['y_pred'] = results['y_pred']
    test_data['error'] = test_data['y_pred'] - test_data['y_true']
    test_data['abs_error'] = np.abs(test_data['error'])
    
    # 按时间段聚合指标
    period_metrics = test_data.groupby('time_period').agg({
        'abs_error': ['mean', 'std'],
        'error': 'mean',
        'y_true': 'mean',
        'y_pred': 'mean'
    }).round(3)
    
    # 绘制
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    periods = period_metrics.index.tolist()
    period_names = {
        'morning_peak': '早高峰',
        'lunch_peak': '午高峰',
        'day_off_peak': '白天平峰',
        'dinner_peak': '晚高峰',
        'night': '夜间'
    }
    period_labels = [period_names.get(p, p) for p in periods]
    
    # 1. MAE按时间段
    ax = axes[0, 0]
    mae_values = period_metrics[('abs_error', 'mean')]
    ax.bar(period_labels, mae_values, color='steelblue', alpha=0.7)
    ax.set_ylabel('平均绝对误差 (MAE)', fontsize=11)
    ax.set_title('不同时间段的预测误差', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. 平均压力分数（真实vs预测）
    ax = axes[0, 1]
    x = np.arange(len(periods))
    width = 0.35
    ax.bar(x - width/2, period_metrics[('y_true', 'mean')], width, 
           label='真实值', alpha=0.7)
    ax.bar(x + width/2, period_metrics[('y_pred', 'mean')], width,
           label='预测值', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, rotation=45, ha='right')
    ax.set_ylabel('平均压力分数', fontsize=11)
    ax.set_title('不同时间段的平均压力分数', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 误差箱线图
    ax = axes[1, 0]
    error_data = [test_data[test_data['time_period'] == p]['error'].values 
                  for p in periods]
    bp = ax.boxplot(error_data, labels=period_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_ylabel('预测误差', fontsize=11)
    ax.set_title('不同时间段的误差分布', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. R²按时间段
    ax = axes[1, 1]
    r2_values = []
    for period in periods:
        period_data = test_data[test_data['time_period'] == period]
        if len(period_data) > 1:
            r2 = r2_score(period_data['y_true'], period_data['y_pred'])
            r2_values.append(r2)
        else:
            r2_values.append(0)
    
    ax.bar(period_labels, r2_values, color='green', alpha=0.7)
    ax.set_ylabel('R² 分数', fontsize=11)
    ax.set_title('不同时间段的R²分数', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 保存时间段分析图: {save_path}")
    
    return period_metrics


def main():
    print("[INFO] 开始验证预测模型...")
    
    if not XGB_AVAILABLE:
        print("[ERROR] XGBoost未安装，无法验证模型")
        print("[INFO] 请先安装: pip install xgboost")
        return
    
    # 1. 加载数据
    processed_dir = os.path.join("data", "processed", "preprocessed_data(28_days)")
    hierarchical = "rider_stress_hierarchical_final.csv"
    xgb_opt = "xgb_optimized_stress.csv"
    
    print("[INFO] 加载数据集...")
    ds = prepare_dataset(
        processed_dir,
        hierarchical_file=hierarchical,
        xgb_file=xgb_opt,
        window=3,
        horizon=1,
        target_metric="stress_score__day_off_peak",
    )
    
    if ds is None or ds['X'].size == 0:
        print("[ERROR] 数据集为空")
        return
    
    X = ds['X']
    y = ds['y']
    meta = ds['meta']
    pivot_df = ds['pivot']
    
    print(f"[INFO] 数据集大小: X={X.shape}, y={y.shape}")
    
    # 2. 时间序列分割
    print("[INFO] 进行时间序列分割...")
    (X_train, y_train, meta_train), (X_test, y_test, meta_test) = \
        time_series_split(X, y, meta, train_ratio=0.7)
    
    print(f"[INFO] 训练集: {len(X_train)} 样本")
    print(f"[INFO] 测试集: {len(X_test)} 样本")
    
    # 3. 加载模型（注意：当前模型是用所有数据训练的，这里重新训练以适应验证）
    print("[INFO] 在训练集上重新训练模型（用于验证）...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # 4. 在测试集上预测
    print("[INFO] 在测试集上预测...")
    results = validate_predictions(model, X_test, y_test)
    
    # 5. 打印指标
    print("\n" + "="*50)
    print("模型验证结果")
    print("="*50)
    print("\n回归指标:")
    for key, value in results['regression'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n分类指标（压力等级准确率）:")
    for key, value in results['classification'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n误差统计:")
    for key, value in results['error_stats'].items():
        print(f"  {key}: {value:.4f}")
    
    # 6. 可视化
    print("\n[INFO] 生成可视化图表...")
    plot_predictions_vs_actual(results['y_true'], results['y_pred'],
                              OUTPUT_DIR / 'model_validation_predictions.png')
    
    # 7. 按时间段分析（需要原始数据中的time_period信息）
    # 从pivot_df重建time_period信息
    test_dates = [m['target_date'] for m in meta_test]
    test_couriers = [m['courier_id'] for m in meta_test]
    
    # 尝试从pivot_df获取time_period信息
    all_data_with_split = pd.DataFrame({
        'courier_id': test_couriers + [m['courier_id'] for m in meta_train],
        'date': test_dates + [m['target_date'] for m in meta_train],
        'split': ['test'] * len(test_couriers) + ['train'] * len(meta_train),
        'time_period': ['day_off_peak'] * (len(test_couriers) + len(meta_train))  # 默认值
    })
    
    # 如果pivot_df中有日期信息，可以更精确匹配
    plot_performance_by_time_period(all_data_with_split, results,
                                   OUTPUT_DIR / 'model_validation_by_period.png')
    
    # 8. 保存验证报告
    report = {
        'regression_metrics': results['regression'],
        'classification_metrics': results['classification'],
        'error_stats': results['error_stats'],
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    import json
    report_path = OUTPUT_DIR / 'model_validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] 保存验证报告: {report_path}")
    
    print("\n" + "="*50)
    print("模型验证完成！")
    print("="*50)
    
    # 判断模型是否有效
    r2 = results['regression']['R2']
    mae = results['regression']['MAE']
    
    print(f"\n模型效果评估:")
    if r2 > 0.8:
        print(f"  ✅ R² = {r2:.4f} > 0.8，模型解释能力强")
    elif r2 > 0.5:
        print(f"  ⚠️  R² = {r2:.4f}，模型有一定解释能力，但可以改进")
    else:
        print(f"  ❌ R² = {r2:.4f} < 0.5，模型解释能力较弱")
    
    if mae < 1.0:
        print(f"  ✅ MAE = {mae:.4f} < 1.0，预测误差较小")
    elif mae < 2.0:
        print(f"  ⚠️  MAE = {mae:.4f}，预测误差中等")
    else:
        print(f"  ❌ MAE = {mae:.4f} > 2.0，预测误差较大")
    
    print(f"\n结论: 模型{'有效' if r2 > 0.5 and mae < 2.0 else '需要改进'}")


if __name__ == '__main__':
    main()

