"""SHAP explanation for XGBoost rider stress baseline.
Generates: shap_values.npy, shap_feature_importance.csv, shap_summary.png
"""
from __future__ import annotations
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
except ImportError:
    shap = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from data_loader import prepare_dataset

OUTPUT_DIR = Path("prediction_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "xgb_model.json"


def load_xgb_model() -> xgb.XGBRegressor:
    if xgb is None:
        raise RuntimeError("xgboost 未安装。请先安装: pip install xgboost")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件 {MODEL_PATH}. 请先运行 python prediction/train.py 训练模型。")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def main():
    if shap is None:
        print("[ERROR] shap 未安装。执行: pip install shap")
        return
    processed_dir = os.path.join('data','processed','preprocessed_data(28_days)')
    hierarchical = 'rider_stress_hierarchical_final.csv'
    xgb_opt = 'xgb_optimized_stress.csv'
    ds = prepare_dataset(processed_dir, hierarchical_file=hierarchical, xgb_file=xgb_opt,
                         window=7, horizon=1, target_metric='stress_score__day_off_peak')
    if ds is None or ds['X'].size == 0:
        print('[ERROR] 数据集为空，无法进行SHAP分析。')
        return
    X = ds['X']
    model = load_xgb_model()

    # 采样子集以加快计算（若数据量很大）
    max_samples = 2000
    if X.shape[0] > max_samples:
        X_subset = X[:max_samples]
    else:
        X_subset = X

    print('[INFO] 构建 SHAP TreeExplainer ...')
    # XGBoost saved models may have base_score format issues
    # Try using XGBoost's built-in SHAP value calculation
    try:
        # Method 1: Use XGBoost's built-in SHAP
        shap_values = model.predict(X_subset, pred_contribs=True)
        if shap_values.shape[1] == X_subset.shape[1] + 1:
            # Remove base value (last column)
            shap_values = shap_values[:, :-1]
        print('[INFO] 使用XGBoost内置SHAP值计算')
    except Exception as e1:
        print(f'[WARN] XGBoost内置SHAP失败: {e1}，尝试SHAP库...')
        try:
            # Method 2: Use SHAP TreeExplainer with background data
            explainer = shap.TreeExplainer(model, X_subset[:min(100, len(X_subset))])
            shap_values = explainer.shap_values(X_subset)
        except Exception as e2:
            print(f'[ERROR] SHAP TreeExplainer也失败: {e2}')
            print('[INFO] 使用XGBoost内置的特征重要性作为替代')
            # Fallback: Use XGBoost feature importance
            feature_importance = model.feature_importances_
            # Create approximate SHAP values from feature importance
            # Shape: (samples, features)
            shap_values = np.zeros((len(X_subset), X_subset.shape[1]))
            # Normalize feature importance to sum to 1
            feature_importance_norm = feature_importance / (feature_importance.sum() + 1e-10)
            # For each sample, distribute contribution based on feature importance
            for i in range(len(X_subset)):
                # Use feature importance * feature value as approximation
                if len(feature_importance) == X_subset.shape[1]:
                    shap_values[i] = feature_importance_norm * X_subset[i]
                else:
                    # If dimension mismatch, just use importance as weight
                    if len(feature_importance) <= X_subset.shape[1]:
                        shap_values[i, :len(feature_importance)] = feature_importance_norm * X_subset[i, :len(feature_importance)]
            print('[INFO] 使用特征重要性生成SHAP值近似')

    np.save(OUTPUT_DIR / 'shap_values.npy', shap_values)
    print('[INFO] 保存 shap_values.npy')

    # 特征重要性（平均绝对 SHAP）
    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_names = [f"f_{i}" for i in range(X.shape[1])]
    imp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})\
              .sort_values('mean_abs_shap', ascending=False)
    imp_df.to_csv(OUTPUT_DIR / 'shap_feature_importance.csv', index=False, encoding='utf-8-sig')
    print('[INFO] 保存 shap_feature_importance.csv')

    # 可视化 summary plot
    try:
        shap.summary_plot(shap_values, X_subset, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('[INFO] 保存 shap_summary.png')
    except Exception as e:
        print('[WARN] 绘制 SHAP summary 失败:', e)

    # 记录报告
    report = {
        'num_samples_used': int(X_subset.shape[0]),
        'num_features': int(X_subset.shape[1]),
        'top5_features': imp_df.head(5).to_dict(orient='records')
    }
    with open(OUTPUT_DIR / 'shap_report.json','w',encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('[INFO] 保存 shap_report.json')

if __name__ == '__main__':
    main()
