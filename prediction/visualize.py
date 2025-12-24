"""
骑手配送压力预测 - 可视化模块
用于绘制 Bi-LSTM 模型的预测结果，包括：
1. 真实值 vs 预测值散点图
2. 典型骑手的时序预测曲线
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch not installed")
    exit(1)

OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ==================== Bi-LSTM 模型定义 ====================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.bilstm(x)
        last_hidden = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        return self.fc(last_hidden)


# ==================== 数据加载 ====================
def load_preprocessed_data(data_dir):
    """加载预处理后的数据"""
    pattern = os.path.join(data_dir, "rider_features_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")
    
    print(f"[DATA] Found {len(files)} daily feature files")
    
    df_list = []
    for f in files:
        df = pd.read_csv(f, encoding='utf-8')
        date_str = os.path.basename(f).replace('rider_features_', '').replace('.csv', '')
        df['date'] = date_str
        df_list.append(df)
    
    data = pd.concat(df_list, ignore_index=True)
    print(f"[DATA] Total samples: {len(data)}")
    print(f"[DATA] Unique couriers: {data['courier_id'].nunique()}")
    
    return data


def create_time_lag_features(data, lag_steps=3):
    """创建时间滑窗特征"""
    data_sorted = data.sort_values(['courier_id', 'date']).reset_index(drop=True)
    
    lag_features = []
    lead_feature = []
    valid_indices = []
    
    for courier_id, group_df in data_sorted.groupby('courier_id'):
        group_indices = group_df.index.tolist()
        
        for i in range(lag_steps, len(group_df) - 1):
            current_idx = group_indices[i]
            row_lags = {'index': current_idx}
            
            for lag in range(1, lag_steps + 1):
                past_idx = group_indices[i - lag]
                past_row = data.loc[past_idx]
                
                lag_features_list = [
                    'order_rate', 'avg_speed', 'continuous_orders', 'load_intensity',
                    'task_density', 'congestion_index', 'weather_score', 'task_per_km',
                    'avg_interval', 'dsi'
                ]
                
                for feat in lag_features_list:
                    if feat in past_row.index:
                        row_lags[f'lag_{lag}_{feat}'] = past_row[feat]
            
            future_idx = group_indices[i + 1]
            future_dsi = data.loc[future_idx, 'dsi']
            
            lag_features.append(row_lags)
            lead_feature.append(future_dsi)
            valid_indices.append(current_idx)
    
    if not lag_features:
        return pd.DataFrame()

    lag_df = pd.DataFrame(lag_features).set_index('index')
    data_lagged = data.loc[valid_indices].copy()
    data_lagged['dsi_target'] = lead_feature
    
    data_lagged = data_lagged.join(lag_df)
    
    return data_lagged


def prepare_features_and_target_no_leakage(data_lagged):
    """准备无泄露的特征和目标"""
    base_features = [
        'order_rate', 'avg_speed', 'continuous_orders', 
        'load_intensity', 'task_density', 'congestion_index',
        'weather_score', 'task_per_km', 'avg_interval'
    ]
    
    lag_cols = [col for col in data_lagged.columns if col.startswith('lag_')]
    feature_cols = [col for col in base_features if col in data_lagged.columns]
    feature_cols.extend(lag_cols)
    
    valid_mask = data_lagged['dsi_target'].notna()
    data_valid = data_lagged[valid_mask].copy()
    
    for col in feature_cols:
        if col in data_valid.columns:
            data_valid[col] = data_valid[col].fillna(data_valid[col].median())
    
    X = data_valid[feature_cols].values
    y = data_valid['dsi_target'].values
    
    return data_valid, X, y, feature_cols


def create_bilstm_sequences_panel(data_valid, feature_cols, seq_length=3):
    """为面板数据创建序列"""
    X_seq, y_seq, courier_seq_ids = [], [], []
    
    for courier_id, group_df in data_valid.groupby('courier_id'):
        group_df = group_df.sort_values('date')
        X_courier = group_df[feature_cols].values
        y_courier = group_df['dsi_target'].values
        
        if len(X_courier) < seq_length: 
            continue
        
        for i in range(seq_length - 1, len(X_courier)):
            window = X_courier[i - seq_length + 1 : i + 1]
            target = y_courier[i]
            
            X_seq.append(window)
            y_seq.append(target)
            courier_seq_ids.append(courier_id)
            
    return np.array(X_seq), np.array(y_seq), np.array(courier_seq_ids)


# ==================== 推理函数 ====================
def predict_in_batches(model, tensor_data, batch_size=1024, device=torch.device('cpu')):
    """分批推理以避免内存溢出"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(tensor_data), batch_size):
            end_idx = min(i + batch_size, len(tensor_data))
            batch = tensor_data[i:end_idx].to(device)
            batch_pred = model(batch)
            predictions.append(batch_pred.cpu().numpy())
    
    return np.vstack(predictions).flatten()


# ==================== 可视化函数 ====================
def plot_scatter_true_vs_pred(y_true, y_pred, save_path=None):
    """
    绘制真实值 vs 预测值散点图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # 对角线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')
    
    # 计算 R2
    r2 = r2_score(y_true, y_pred)
    
    # 标签和标题
    ax.set_xlabel('True DSI (Delivery Stress Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted DSI', fontsize=12, fontweight='bold')
    ax.set_title(f'Bi-LSTM Model: True vs Predicted DSI\n(R² = {r2:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Scatter plot saved to {save_path}")
    
    plt.close()


def plot_timeseries_single_courier(data_valid, y_true, y_pred, courier_seq_ids, seq_length=3, save_path=None):
    """
    绘制典型骑手的时序预测曲线
    """
    # 随机选取一个骑手
    unique_couriers = np.unique(courier_seq_ids)
    selected_courier = np.random.choice(unique_couriers)
    
    # 获取该骑手的数据
    courier_mask = courier_seq_ids == selected_courier
    courier_indices = np.where(courier_mask)[0]
    
    y_true_courier = y_true[courier_mask]
    y_pred_courier = y_pred[courier_mask]
    
    # 获取该骑手的日期信息用于 x 轴标签
    courier_data = data_valid[data_valid['courier_id'] == selected_courier].sort_values('date')
    dates = courier_data['date'].values
    
    # 创建时间索引（序列索引）
    # 由于序列是通过滑动窗口创建的，我们需要对应正确的时间点
    time_indices = np.arange(len(y_true_courier))
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(time_indices, y_true_courier, 'o-', color='red', linewidth=2.5, 
            markersize=8, label='True DSI', alpha=0.8)
    ax.plot(time_indices, y_pred_courier, 's--', color='blue', linewidth=2.5, 
            markersize=8, label='Predicted DSI', alpha=0.8)
    
    # 填充两条线之间的区域以显示误差
    ax.fill_between(time_indices, y_true_courier, y_pred_courier, alpha=0.15, color='gray')
    
    # 标签和标题
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('DSI (Delivery Stress Index)', fontsize=12, fontweight='bold')
    ax.set_title(f'Time-Series Prediction for Courier {selected_courier}\n(Total sequences shown)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Time-series plot saved to {save_path}")
    
    plt.close()
    
    print(f"[VIZ] Selected courier: {selected_courier}, sequences count: {len(y_true_courier)}")


def main():
    print("\n" + "="*80)
    print("骑手配送压力预测 - 可视化模块")
    print("="*80 + "\n")
    
    # 1. 加载数据
    print("[STEP 1] Loading data...")
    data_dir = "./output_features"
    if not os.path.exists(data_dir):
        print("[ERROR] Data dir not found")
        return
    
    data = load_preprocessed_data(data_dir)
    
    print("\n[STEP 2] Creating time-lagged features...")
    data_lagged = create_time_lag_features(data, lag_steps=3)
    
    print("\n[STEP 3] Preparing features and targets...")
    data_valid, X, y, feature_cols = prepare_features_and_target_no_leakage(data_lagged)
    
    if len(X) < 100:
        print(f"[ERROR] Insufficient samples ({len(X)})")
        return
    
    # 2. 特征归一化
    print("\n[STEP 4] Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    data_valid_scaled = data_valid.copy()
    data_valid_scaled[feature_cols] = X_scaled
    
    # 3. 创建序列
    print("\n[STEP 5] Creating LSTM sequences...")
    X_seq, y_seq, courier_seq_ids = create_bilstm_sequences_panel(data_valid_scaled, feature_cols, seq_length=3)
    print(f"[DATA] Total sequences: {X_seq.shape[0]}")
    print(f"[DATA] Unique couriers: {len(np.unique(courier_seq_ids))}")
    
    # 4. 加载模型
    print("\n[STEP 6] Loading trained Bi-LSTM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MODEL] Device: {device}")
    
    model = BiLSTMModel(input_size=len(feature_cols), hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load(OUTPUT_DIR / "bilstm_best.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # 5. 全量推理
    print("\n[STEP 7] Running inference on all sequences...")
    X_seq_t = torch.FloatTensor(X_seq).to(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    y_pred = predict_in_batches(model, X_seq_t, batch_size=1024, device=device)
    
    print(f"[PRED] Predictions shape: {y_pred.shape}")
    print(f"[PRED] True values shape: {y_seq.shape}")
    
    # 6. 绘制可视化
    print("\n[STEP 8] Creating visualizations...")
    
    # 散点图
    plot_scatter_true_vs_pred(
        y_seq, y_pred,
        save_path=str(FIGURES_DIR / "01_scatter_true_vs_pred.png")
    )
    
    # 时序图
    plot_timeseries_single_courier(
        data_valid_scaled, y_seq, y_pred, courier_seq_ids,
        seq_length=3,
        save_path=str(FIGURES_DIR / "02_timeseries_single_courier.png")
    )
    
    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print(f"✓ Figures saved to: {FIGURES_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
