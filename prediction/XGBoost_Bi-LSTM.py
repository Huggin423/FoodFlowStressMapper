"""
骑手配送压力预测 - 对比实验框架 (核心逻辑修正版)

【修正说明】
1. 关键修复: 在输入特征中加入了目标订单的已知属性(距离、天气等)，解决了模型"盲猜"的问题。
2. 保持原数据结构和模型参数不变。
"""
import os
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] tqdm not installed. Run: pip install tqdm")
    tqdm = lambda x, **kwargs: x

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] XGBoost not installed")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== 数据加载与时间滑窗特征构造 ====================
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
    return data


def create_time_lag_features(data, lag_steps=3):
    """创建时间滑窗特征 + 添加目标订单的上下文特征"""
    # 按courier+date排序
    data_sorted = data.sort_values(['courier_id', 'date']).reset_index(drop=True)
    
    lag_features = []
    lead_feature = []
    valid_indices = []
    
    # 1. 保留这里的 acc_...，它们会生成 lag_1_acc_time 等特征
    # 这是合法的"接单前疲劳度"
    base_lag_list = [
        'order_rate', 'avg_speed', 'continuous_orders', 'load_intensity',
        'task_density', 'congestion_index', 'weather_score', 'task_per_km',
        'avg_interval', 'dsi',
        'acc_orders_today', 'acc_distance_today', 'acc_time_today' 
    ]

    for courier_id, group_df in tqdm(data_sorted.groupby('courier_id'), desc="Creating lags"):
        group_indices = group_df.index.tolist()
        
        for i in range(lag_steps, len(group_df) - 1):
            current_idx = group_indices[i]
            row_lags = {'index': current_idx}
            
            # --- 收集历史滞后特征 (Lags) ---
            for lag in range(1, lag_steps + 1):
                past_idx = group_indices[i - lag]
                past_row = data.loc[past_idx]
                
                for feat in base_lag_list:
                    if feat in past_row.index:
                        row_lags[f'lag_{lag}_{feat}'] = past_row[feat]
            
            # --- 收集目标订单的"已知"属性 (Context) ---
            future_idx = group_indices[i + 1]
            future_row = data.loc[future_idx]
            
            # 基础物理属性 (这些是合法的)
            row_lags['target_r_dis_all'] = future_row['r_dis_all']
            row_lags['target_task_per_km'] = future_row['task_per_km']
            row_lags['target_weather'] = future_row['weather_score']
            row_lags['target_max_load'] = future_row['max_load']
            
            # 3. 收集预测目标 (Target)
            future_dsi = future_row['dsi']
            
            lag_features.append(row_lags)
            lead_feature.append(future_dsi)
            valid_indices.append(current_idx)
    
    if not lag_features:
        return pd.DataFrame()

    lag_df = pd.DataFrame(lag_features).set_index('index')
    data_lagged = data.loc[valid_indices].copy()
    data_lagged['dsi_target'] = lead_feature
    
    # 合并
    data_lagged = data_lagged.join(lag_df)
    
    return data_lagged


def prepare_features_and_target_no_leakage(data_lagged):
    """
    准备无泄露的特征和目标
    """
    # 基础特征（当前时刻 i 的状态）
    # 【新增】当前时刻的累积状态
    base_features = [
        'order_rate', 'avg_speed', 'continuous_orders', 
        'load_intensity', 'task_density', 'congestion_index',
        'weather_score', 'task_per_km', 'avg_interval',
        'acc_orders_today', 'acc_distance_today', 'acc_time_today' # <--- 新增
    ]
    
    # 滞后特征 (自动捕获 lag_ 开头的列)
    lag_cols = [col for col in data_lagged.columns if col.startswith('lag_')]
    
    # 目标上下文特征 (i+1 的已知属性)
    # 【新增】目标时刻的累积疲劳
    target_context_cols = [
        'target_r_dis_all', 'target_task_per_km', 
        'target_weather', 'target_max_load'
    ]
    
    # 合并所有特征列
    feature_cols = [col for col in base_features if col in data_lagged.columns]
    feature_cols.extend(lag_cols)
    feature_cols.extend([col for col in target_context_cols if col in data_lagged.columns])
    
    # ... (后续代码不变) ...
    
    # 过滤无效数据
    valid_mask = data_lagged['dsi_target'].notna()
    data_valid = data_lagged[valid_mask].copy()
    
    # 填充缺失值
    for col in feature_cols:
        if col in data_valid.columns:
            data_valid[col] = data_valid[col].fillna(data_valid[col].median())
    
    X = data_valid[feature_cols].values
    y = data_valid['dsi_target'].values
    
    print(f"[DATA] Feature matrix shape: {X.shape}")
    print(f"[DATA] Features used: {len(feature_cols)}")
    print(f"[DATA] Added Cumulative Context: {['target_acc_distance', 'target_acc_orders', 'target_acc_time']}")
    
    return data_valid, X, y, feature_cols


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}


def predict_in_batches(model, tensor_data, batch_size=1024, device=torch.device('cpu')):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(tensor_data), batch_size):
            end_idx = min(i + batch_size, len(tensor_data))
            batch = tensor_data[i:end_idx].to(device)
            batch_pred = model(batch)
            predictions.append(batch_pred.cpu().numpy())
    return np.vstack(predictions).flatten()


# ==================== Baseline: XGBoost ====================
def train_xgboost_baseline(X_train, X_val, y_train, y_val, feature_cols):
    if not XGB_AVAILABLE: return None
    print("\n" + "="*70)
    print("BASELINE MODEL: XGBoost Regression")
    print("="*70)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=200, max_depth=6,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    train_metrics = calculate_metrics(y_train, model.predict(X_train))
    val_metrics = calculate_metrics(y_val, model.predict(X_val))
    
    print(f"[XGBoost] Train R^2: {train_metrics['R2']:.4f}")
    print(f"[XGBoost] Val   R^2: {val_metrics['R2']:.4f}")
    
    return {'train_metrics': train_metrics, 'val_metrics': val_metrics}


# ==================== Bi-LSTM Model ====================
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


def create_bilstm_sequences_panel(data_valid, feature_cols, seq_length=10):
    X_seq, y_seq, courier_seq_ids = [], [], []
    for courier_id, group_df in data_valid.groupby('courier_id'):
        group_df = group_df.sort_values('date')
        X_courier = group_df[feature_cols].values
        y_courier = group_df['dsi_target'].values
        
        if len(X_courier) < seq_length: continue
        
        for i in range(seq_length - 1, len(X_courier)):
            window = X_courier[i - seq_length + 1 : i + 1]
            target = y_courier[i]
            X_seq.append(window)
            y_seq.append(target)
            courier_seq_ids.append(courier_id)
            
    return np.array(X_seq), np.array(y_seq), np.array(courier_seq_ids)


def train_bilstm_model(data_valid, feature_cols, seq_length=3, hidden_size=128):
    if not TORCH_AVAILABLE: return None
    print("\n" + "="*70)
    print("PROPOSED MODEL: Bi-LSTM")
    print("="*70)
    
    X_seq, y_seq, _ = create_bilstm_sequences_panel(data_valid, feature_cols, seq_length)
    
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=True, random_state=42
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.FloatTensor(X_train_s).to(device)
    X_val_t = torch.FloatTensor(X_val_s).to(device)
    y_train_t = torch.FloatTensor(y_train_s).unsqueeze(1).to(device)
    y_val_t = torch.FloatTensor(y_val_s).unsqueeze(1).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
    
    model = BiLSTMModel(input_size=len(feature_cols), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_r2 = -float('inf')
    patience, counter = 15, 0
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_r2 = r2_score(y_val_s, val_pred.cpu().numpy())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val R^2: {val_r2:.4f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), OUTPUT_DIR / "bilstm_best.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(OUTPUT_DIR / "bilstm_best.pt"))
    model.eval()
    if device.type == 'cuda': torch.cuda.empty_cache()
    
    val_pred = predict_in_batches(model, X_val_t, device=device)
    return {'val_metrics': calculate_metrics(y_val_s, val_pred)}


def main():
    print("\n" + "="*80 + "\n骑手配送压力预测\n" + "="*80)
    data_dir = "./output_features"
    if not os.path.exists(data_dir):
        print("[ERROR] Data dir not found")
        return

    data = load_preprocessed_data(data_dir)
    print("\n[FEATURE] Creating time-lagged features...")
    data_lagged = create_time_lag_features(data, lag_steps=3)
    
    data_valid, X, y, feature_cols = prepare_features_and_target_no_leakage(data_lagged)
    
    if len(X) < 100: return
    
    print("\n[NORMALIZATION] StandardScaler fit_transform...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    data_valid_scaled = data_valid.copy()
    data_valid_scaled[feature_cols] = X_scaled
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {'models': {}}
    if XGB_AVAILABLE:
        results['models']['XGBoost'] = train_xgboost_baseline(X_train, X_val, y_train, y_val, feature_cols)
    if TORCH_AVAILABLE:
        results['models']['Bi-LSTM'] = train_bilstm_model(data_valid_scaled, feature_cols)

if __name__ == "__main__":
    main()