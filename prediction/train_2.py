"""
骑手配送压力预测 - 进阶版 v2.1 (Transformer + 动态坐标提取 + 全局路网融合)

【核心修复】
1. 坐标回填: 自动读取原始 GeoJSON (DeliveryRoutes) 提取骑手每日的中心经纬度。
2. 全局路网: 支持读取文件夹下所有的 OSM JSON 文件，建立全局网格索引，解决地图覆盖不全的问题。
3. 日期对齐: 修复了 GeoJSON (int 20200201) 与 CSV (str '20200201') 的匹配问题。
"""

import os
import json
import glob
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

# 检查设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Running on {DEVICE}")

OUTPUT_DIR = Path("outputs_transformer")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 1. 坐标提取器 (新增模块) ====================
class CoordinateExtractor:
    """
    负责从原始 GeoJSON 数据中提取 (courier_id, date) -> (center_lat, center_lon)
    """
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.coords_map = {} # (courier_id, date_str) -> (lat, lon)

    def process(self):
        # 假设原始文件名为 DeliveryRoutes_*\.json 或 *\.geojson
        abs_search_path = os.path.abspath(self.raw_data_dir)
        print(f"[Coordinate] Searching in: {abs_search_path}")
        if not os.path.isdir(self.raw_data_dir):
            print(f"[ERROR] Raw data directory does not exist: {abs_search_path}")
            return

        json_files = glob.glob(os.path.join(self.raw_data_dir, "*.json"))
        geojson_files = glob.glob(os.path.join(self.raw_data_dir, "*.geojson"))
        files = sorted(json_files + geojson_files)
        
        if not files:
            print(f"[WARN] No raw JSON/GeoJSON files found in {abs_search_path}. Cannot extract coordinates.")
            return

        print(f"[Coordinate] Extracting locations from {len(files)} raw files...")
        count = 0
        
        for f_path in tqdm(files, desc="Parsing GeoJSON"):
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                features = data.get('features', [])
                for feat in features:
                    props = feat.get('properties', {})
                    geom = feat.get('geometry', {})
                    
                    # 只要 Route 类型的数据，因为它包含完整的路径点
                    if props.get('feature_type') != 'route':
                        continue
                        
                    c_id = props.get('courier_id')
                    date_val = props.get('date') # 可能是 int 20200201
                    
                    if c_id is None or date_val is None or geom is None:
                        continue
                        
                    # 统一类型为字符串，确保与CSV中的字符串日期对齐
                    date_str = str(date_val)
                    
                    # 提取坐标 (GeoJSON 格式为 [lon, lat])
                    coords = geom.get('coordinates', [])
                    if not coords:
                        continue
                    
                    # 转换为 numpy 计算中心点
                    coords_np = np.array(coords)
                    # 注意: GeoJSON 是 [lon, lat]，我们需要 [lat, lon]
                    lons = coords_np[:, 0]
                    lats = coords_np[:, 1]
                    
                    mean_lat = np.mean(lats)
                    mean_lon = np.mean(lons)
                    
                    self.coords_map[(str(c_id), date_str)] = (mean_lat, mean_lon)
                    count += 1
                    
            except Exception as e:
                print(f"[WARN] Error reading {f_path}: {e}")
        
        print(f"[Coordinate] Extracted coordinates for {len(self.coords_map)} unique routes.")

    def get_coords(self, courier_id, date):
        return self.coords_map.get((str(courier_id), str(date)), (np.nan, np.nan))

# ==================== 2. 路网特征处理器 (支持多文件) ====================
class RoadNetworkEmbedder:
    """
    负责解析 OSM JSON 数据，支持加载目录下所有 JSON 文件
    """
    def __init__(self, map_dir_or_file, grid_precision=0.001):
        self.path = map_dir_or_file
        self.grid_precision = grid_precision
        self.grid_features = defaultdict(int) 
        self.loaded = False

    def load_and_process(self):
        if os.path.isdir(self.path):
            files = glob.glob(os.path.join(self.path, "*.json"))
        else:
            files = [self.path]
            
        print(f"\n[RoadNetwork] Loading {len(files)} map files...")
        total_nodes = 0
        
        for json_path in tqdm(files, desc="Processing Map Nodes"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    osm_data = json.load(f)
                
                # 支持两种结构：字典包含 elements，或直接是列表
                if isinstance(osm_data, list):
                    elements = osm_data
                else:
                    elements = osm_data.get('elements', [])
                for elem in elements:
                    if elem['type'] == 'node' and 'lat' in elem and 'lon' in elem:
                        lat_idx = int(elem['lat'] / self.grid_precision)
                        lon_idx = int(elem['lon'] / self.grid_precision)
                        self.grid_features[(lat_idx, lon_idx)] += 1
                        total_nodes += 1
            except Exception as e:
                print(f"[WARN] Failed to load {json_path}: {e}")

        self.loaded = True
        print(f"[RoadNetwork] Processed {total_nodes} nodes into {len(self.grid_features)} grids.")

    def get_features(self, lat_series, lon_series):
        if not self.loaded:
            return np.zeros(len(lat_series))
        
        features = []
        lats = lat_series.values
        lons = lon_series.values
        
        for lat, lon in zip(lats, lons):
            if pd.isna(lat) or pd.isna(lon) or lat == 0:
                features.append(0)
                continue
            lat_idx = int(lat / self.grid_precision)
            lon_idx = int(lon / self.grid_precision)
            features.append(self.grid_features.get((lat_idx, lon_idx), 0))
            
        return np.array(features)

# ==================== 3. 数据处理流水线 ====================
def load_data_pipeline(feature_dir, raw_data_dir, map_path):
    # 1. 加载特征 CSV
    print("[1/4] Loading feature CSVs...")
    pattern = os.path.join(feature_dir, "rider_features_*.csv")
    files = sorted(glob.glob(pattern))
    if not files: raise FileNotFoundError("No feature CSVs found.")
    
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        date_str = os.path.basename(f).replace('rider_features_', '').replace('.csv', '')
        df['date'] = date_str
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)
    
    # 2. 坐标回填 (如果 CSV 里没有经纬度)
    if 'center_lat' not in data.columns:
        print("[2/4] Coordinates missing. Extracting from raw GeoJSON...")
        extractor = CoordinateExtractor(raw_data_dir)
        extractor.process()
        
        # 将提取的坐标 map 回 dataframe
        # 使用 apply 会比较慢，改用 map
        data['temp_key'] = list(zip(data['courier_id'].astype(str), data['date'].astype(str)))
        
        # 构建快速查询字典
        lat_dict = {k: v[0] for k, v in extractor.coords_map.items()}
        lon_dict = {k: v[1] for k, v in extractor.coords_map.items()}
        
        data['center_lat'] = data['temp_key'].map(lat_dict)
        data['center_lon'] = data['temp_key'].map(lon_dict)
        data.drop(columns=['temp_key'], inplace=True)
        
        missing = data['center_lat'].isna().sum()
        print(f"      Matched coordinates for {len(data) - missing} rows. Missing: {missing}")
        
        # 填充缺失值 (用均值填充，防止报错)
        data['center_lat'].fillna(data['center_lat'].mean(), inplace=True)
        data['center_lon'].fillna(data['center_lon'].mean(), inplace=True)
    
    # 3. 路网融合
    print("[3/4] Fusing Road Network features...")
    embedder = RoadNetworkEmbedder(map_path)
    embedder.load_and_process()
    
    road_density = embedder.get_features(data['center_lat'], data['center_lon'])
    data['road_node_density'] = np.log1p(road_density) # Log变换处理长尾分布
    
    print(f"      Road Density (Mean): {data['road_node_density'].mean():.4f}")

    # 4. 验证关键特征是否有效，避免在垃圾特征上训练
    center_lat_missing = int(data['center_lat'].isna().sum()) if 'center_lat' in data.columns else len(data)
    road_density_mean = float(data['road_node_density'].mean()) if 'road_node_density' in data.columns else 0.0
    if road_density_mean == 0.0 or center_lat_missing > 0:
        raise ValueError(
            f"[STOP] Invalid fused features: road_node_density.mean={road_density_mean:.4f}, "
            f"center_lat_missing={center_lat_missing}. Please fix raw data or paths before training."
        )
    
    return data

def create_panel_dataset(data, seq_length=3):
    """创建时序样本 (Batch, Seq_Len, Features)"""
    print("[4/4] Creating sequences...")
    
    feature_cols = [
        'order_rate', 'avg_speed', 'continuous_orders', 'load_intensity',
        'task_density', 'congestion_index', 'weather_score', 
        'road_node_density', # 新增路网特征
        'dsi' # 自回归特征
    ]
    
    # 归一化 (RobustScaler 对异常值更鲁棒)
    scaler = RobustScaler()
    data_scaled = data.copy()
    
    # 填充 NaN
    for col in feature_cols:
        if col in data_scaled.columns:
            data_scaled[col] = data_scaled[col].fillna(data_scaled[col].median())
            
    data_scaled[feature_cols] = scaler.fit_transform(data_scaled[feature_cols])
    
    X_seq, y_seq = [], []
    
    for _, group in tqdm(data_scaled.groupby('courier_id')):
        group = group.sort_values('date')
        vals = group[feature_cols].values
        targets = group['dsi'].values 
        
        if len(vals) < seq_length + 1: continue
            
        for i in range(len(vals) - seq_length):
            window = vals[i : i+seq_length]
            target = targets[i + seq_length] # 预测 T+1
            X_seq.append(window)
            y_seq.append(target)
            
    return np.array(X_seq), np.array(y_seq), len(feature_cols)

# ==================== 4. Transformer 模型 ====================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) # 简化版位置编码
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, seq, feat]
        x = self.embedding(x) * math.sqrt(x.size(2))
        # Add position encoding (broadcasting)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        x = self.transformer_encoder(x)
        
        # Pooling: 只取最后一个时间步，或者平均
        x = x.mean(dim=1) 
        return self.decoder(x)

# ==================== 5. 训练主流程 ====================
def train_model(X, y, input_dim):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)
    
    model = TimeSeriesTransformer(input_size=input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print(f"\n[Training] Start Transformer training (Input Dim: {input_dim})...")
    best_r2 = -float('inf')
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            # Batch processing for validation to avoid OOM
            val_preds = []
            batch_size_val = 1024
            for i in range(0, len(X_val_t), batch_size_val):
                batch = X_val_t[i : i+batch_size_val]
                val_preds.append(model(batch).cpu().numpy())
            
            val_pred_all = np.vstack(val_preds)
            val_r2 = r2_score(y_val, val_pred_all)
            
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), OUTPUT_DIR / "transformer_best.pt")
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val R^2: {val_r2:.4f}")
            
    print(f"[Result] Best Validation R^2: {best_r2:.4f}")
    return best_r2

# ==================== Main ====================
def main():
    # 配置路径
    FEATURE_DIR = "output_features" # 你的特征CSV目录
    RAW_DATA_DIR = "data/raw/ODIDMob_Routes"  # 【重要】请修改为你存放 DeliveryRoutes_*.json 的目录
    MAP_PATH = "data/raw/map"       # 【重要】请修改为你存放 OSM JSON 地图的目录
    
    # 检查路径是否存在
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] Raw data dir not found: {RAW_DATA_DIR}")
        print("请创建一个目录并将原始 DeliveryRoutes_*.json 文件放入其中，以便提取坐标。")
        return

    # 1. 加载并融合
    data = load_data_pipeline(FEATURE_DIR, RAW_DATA_DIR, MAP_PATH)
    
    # 2. 构建序列
    X, y, input_dim = create_panel_dataset(data)
    
    if len(X) == 0:
        print("[ERROR] No data samples created.")
        return
        
    # 3. 训练
    train_model(X, y, input_dim)

if __name__ == "__main__":
    main()