"""
骑手配送压力预测 - STGCN (Spatio-Temporal Graph Convolutional Network) - 最终版 (小时级聚合)
==========================================================================================
功能：
1. 将骑手数据映射到城市网格 (Grid Mapping)
2. 构建基于距离的网格邻接矩阵 (Distance-based Adjacency Matrix)
3. 训练 T-GCN 模型 (Temporal GCN = GCN + GRU)
4. [核心优化] 使用小时级数据 (Hour-level) 进行时空聚合，大幅增加样本量
"""

import os
import json
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

# 配置
CONFIG = {
    'grid_size': 0.01,       # 网格大小 (约1km)
    'seq_len': 6,            # 输入过去6个时间步 (即6小时)
    'pred_len': 1,           # 预测未来1个时间步 (即1小时)
    'hidden_dim': 64,
    'batch_size': 32,
    'epochs': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# ==================== 1. 基础工具类 ====================

class GridSystem:
    """网格系统：将经纬度映射为 Grid ID"""
    def __init__(self, raw_data_dir, grid_size=0.01):
        print("[Grid] Initializing grid system...")
        self.grid_size = grid_size
        self.min_lat, self.max_lat = 90, -90
        self.min_lon, self.max_lon = 180, -180
        
        # 1. 扫描原始数据确定城市边界
        self._scan_boundaries(raw_data_dir)
        
        # 2. 计算网格维度
        self.lat_steps = int((self.max_lat - self.min_lat) / grid_size) + 1
        self.lon_steps = int((self.max_lon - self.min_lon) / grid_size) + 1
        self.num_nodes = self.lat_steps * self.lon_steps
        
        print(f"[Grid] Bounds: Lat[{self.min_lat:.3f}, {self.max_lat:.3f}], Lon[{self.min_lon:.3f}, {self.max_lon:.3f}]")
        print(f"[Grid] Matrix: {self.lat_steps} x {self.lon_steps} = {self.num_nodes} nodes")

    def _scan_boundaries(self, raw_data_dir):
        # 快速扫描部分文件获取大致范围
        files = glob.glob(os.path.join(raw_data_dir, "*.geojson"))[:10] 
        lats, lons = [], []
        for f in files:
            try:
                data = json.load(open(f, 'r', encoding='utf-8'))
                for feat in data.get('features', []):
                    geom = feat.get('geometry', {})
                    if geom and geom.get('coordinates'):
                        coords = np.array(geom['coordinates']).flatten()
                        # GeoJSON是 [lon, lat]
                        lons.extend(coords[0::2])
                        lats.extend(coords[1::2])
            except: pass
        
        if lats:
            self.min_lat, self.max_lat = np.min(lats), np.max(lats)
            self.min_lon, self.max_lon = np.min(lons), np.max(lons)
        else:
            # 默认北京/天津大致范围 (如果扫描失败)
            self.min_lat, self.max_lat = 39.80, 40.10
            self.min_lon, self.max_lon = 116.30, 116.70

    def to_grid_id(self, lat, lon):
        """将经纬度转换为 0 ~ num_nodes-1 的索引"""
        if pd.isna(lat) or pd.isna(lon): return -1
        # 限制在边界内
        lat = min(max(lat, self.min_lat), self.max_lat)
        lon = min(max(lon, self.min_lon), self.max_lon)
        
        row = int((lat - self.min_lat) / self.grid_size)
        col = int((lon - self.min_lon) / self.grid_size)
        return row * self.lon_steps + col


# ==================== 2. 数据处理 ====================

def load_and_aggregate_data(feature_dir, raw_dir):
    """
    1. 读取特征CSV
    2. 从原始GeoJSON回填坐标
    3. 将数据聚合到 Grid
    """
    grid_sys = GridSystem(raw_dir, grid_size=CONFIG['grid_size'])
    
    # --- A. 加载特征 ---
    print("[Data] Loading CSV features...")
    all_files = sorted(glob.glob(os.path.join(feature_dir, "rider_features_*.csv")))
    df_list = []
    for f in all_files:
        temp = pd.read_csv(f)
        # 从文件名提取日期，例如 'rider_features_20200201.csv' -> '20200201'
        temp['date_str'] = os.path.basename(f).replace('rider_features_', '').replace('.csv', '')
        df_list.append(temp)
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # --- B. 坐标回填 (CSV中已有 center_lat/lon) ---
    if 'center_lat' not in raw_df.columns:
        print("[Warn] 'center_lat' not found in CSV. Using random coords (Not Recommended).")
        raw_df['center_lat'] = np.random.uniform(grid_sys.min_lat, grid_sys.max_lat, len(raw_df))
        raw_df['center_lon'] = np.random.uniform(grid_sys.min_lon, grid_sys.max_lon, len(raw_df))
    
    # --- C. 映射到网格 ---
    print("[Data] Mapping to grids...")
    raw_df['grid_id'] = raw_df.apply(lambda x: grid_sys.to_grid_id(x['center_lat'], x['center_lon']), axis=1)
    
    # --- D. 时空聚合 (Key Step for STGCN) ---
    # 【核心修改】：按 [Date + Start_Hour] 聚合，而不是 Time_Period
    print("[Data] Aggregating by [Date + Hour] (High Resolution)...")
    
    # 1. 过滤：只保留比较活跃的 Grid (Top 500)
    grid_activity = raw_df.groupby('grid_id')['dsi'].count()
    top_grids = grid_activity.sort_values(ascending=False).head(500).index.tolist()
    
    # 创建 Grid 映射: 原始Grid ID -> 新的 0~499 索引
    grid_map = {gid: i for i, gid in enumerate(top_grids)}
    raw_df = raw_df[raw_df['grid_id'].isin(top_grids)].copy()
    raw_df['node_idx'] = raw_df['grid_id'].map(grid_map)
    
    # 2. 构建小时级时间步
    # 过滤掉无效时间 (-1)
    if 'start_hour' in raw_df.columns:
        raw_df = raw_df[raw_df['start_hour'] >= 0].copy()
        # 构造类似于 "20200201_09" 的字符串索引
        raw_df['time_step_str'] = raw_df.apply(
            lambda x: f"{x['date_str']}_{int(x['start_hour']):02d}", axis=1
        )
    else:
        print("[Error] 'start_hour' column missing! Falling back to time_period (Low Res).")
        raw_df['time_step_str'] = raw_df['date_str'] + "_" + raw_df['time_period']

    # 排序时间步
    unique_times = sorted(raw_df['time_step_str'].unique())
    time_map = {t: i for i, t in enumerate(unique_times)}
    raw_df['time_idx'] = raw_df['time_step_str'].map(time_map)
    
    num_times = len(unique_times)
    num_nodes = len(top_grids)
    num_feats = 3 # dsi, order_rate, congestion
    
    print(f"[Data] Optimized Granularity: {num_times} Time Steps (Target: ~600+), {num_nodes} Active Nodes")
    
    # 3. 填充矩阵
    agg_df = raw_df.groupby(['time_idx', 'node_idx']).agg({
        'dsi': 'mean', 'order_rate': 'mean', 'congestion_index': 'mean'
    }).reset_index()
    
    data_matrix = np.zeros((num_times, num_nodes, num_feats))
    for _, row in agg_df.iterrows():
        t, n = int(row['time_idx']), int(row['node_idx'])
        data_matrix[t, n, 0] = row['dsi']
        data_matrix[t, n, 1] = row['order_rate']
        data_matrix[t, n, 2] = row['congestion_index']
    
    # ==========================================
    # 4. 重新构建邻接矩阵 (基于距离的真实图)
    # ==========================================
    print("[Graph] Building real adjacency matrix based on distance...")
    
    # 4.1 获取 Top 500 节点的经纬度
    node_coords = raw_df.groupby('node_idx')[['center_lat', 'center_lon']].mean().sort_index()
    coords = node_coords.values # (500, 2)
    
    # 4.2 计算两两距离矩阵 (欧氏距离 approximation)
    lat = coords[:, 0]
    lon = coords[:, 1]
    d_lat = lat[:, np.newaxis] - lat[np.newaxis, :]
    d_lon = lon[:, np.newaxis] - lon[np.newaxis, :]
    dist_sq = d_lat**2 + d_lon**2
    
    # 4.3 构建邻接矩阵 (阈值法)
    # 使用高斯核计算权重 (距离越近权重越大)
    sigma = 0.01
    adj_matrix = np.exp(-dist_sq / (sigma ** 2))
    
    # 稀疏化：只保留权重大的边
    adj_matrix[adj_matrix < 0.1] = 0 
    
    # 归一化 (Row Normalization)
    row_sum = adj_matrix.sum(axis=1)
    row_sum[row_sum == 0] = 1e-6
    adj_norm = adj_matrix / row_sum[:, np.newaxis]
    
    # 转为 Tensor
    adj = torch.FloatTensor(adj_norm)
    
    print(f"[Graph] Adjacency matrix built. Non-zero edges: {(adj > 0).sum()}")

    return data_matrix, adj


def generate_dataset(data, seq_len, pred_len):
    """生成滑窗样本 (B, T, N, F)"""
    X, Y = [], []
    num_times, num_nodes, num_feats = data.shape
    
    for i in range(num_times - seq_len - pred_len + 1):
        # Input: [t, t+seq_len]
        x_seq = data[i : i+seq_len, :, :] 
        # Target: [t+seq_len, t+seq_len+pred_len] 的 DSI (feature 0)
        y_seq = data[i+seq_len : i+seq_len+pred_len, :, 0] 
        
        X.append(x_seq)
        Y.append(y_seq)
        
    return np.array(X), np.array(Y)


# ==================== 3. 模型定义 (T-GCN) ====================

class GraphConv(nn.Module):
    """简单的图卷积层: GCN(X) = A * X * W"""
    def __init__(self, input_dim, output_dim, adj):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.adj = adj # (Nodes, Nodes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: (Batch, Nodes, Features)
        # AX
        x_adj = torch.matmul(self.adj, x) 
        # (AX)W
        out = torch.matmul(x_adj, self.weight) + self.bias
        return out

class TGCNCell(nn.Module):
    """时空图卷积单元 (Graph Conv + GRU Cell)"""
    def __init__(self, adj, input_dim, hidden_dim):
        super(TGCNCell, self).__init__()
        # 这里的 GRU 门控内部使用 GraphConv 而不是 Linear
        self.adj = adj
        self.graph_conv_z = GraphConv(input_dim + hidden_dim, hidden_dim, adj)
        self.graph_conv_r = GraphConv(input_dim + hidden_dim, hidden_dim, adj)
        self.graph_conv_h = GraphConv(input_dim + hidden_dim, hidden_dim, adj)

    def forward(self, x, h):
        # x: (Batch, Nodes, Feat), h: (Batch, Nodes, Hidden)
        cat_input = torch.cat([x, h], dim=2)
        
        z = torch.sigmoid(self.graph_conv_z(cat_input))
        r = torch.sigmoid(self.graph_conv_r(cat_input))
        
        cat_input_new = torch.cat([x, r * h], dim=2)
        h_tilde = torch.tanh(self.graph_conv_h(cat_input_new))
        
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class STGCN_Model(nn.Module):
    def __init__(self, adj, input_dim, hidden_dim, output_dim=1):
        super(STGCN_Model, self).__init__()
        self.tgcn_cell = TGCNCell(adj, input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (Batch, Time, Nodes, Features)
        batch_size, seq_len, num_nodes, _ = x.size()
        
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device)
        
        # 循环处理时间步
        for t in range(seq_len):
            h = self.tgcn_cell(x[:, t, :, :], h)
        
        # Output layer
        out = self.fc(h) # (Batch, Nodes, 1)
        return out


# ==================== 4. 训练流程 ====================

def main():
    print("="*60)
    print("  STGCN Training Pipeline (Hour-Level + Distance Graph)")
    print("="*60)
    
    # 路径配置
    FEATURE_DIR = "./output_features"
    RAW_DIR = "./data/raw/ODIDMob_Routes"
    
    if not os.path.exists(FEATURE_DIR):
        print(f"[Error] Feature dir not found: {FEATURE_DIR}")
        return

    # 1. 准备数据
    data_matrix, adj = load_and_aggregate_data(FEATURE_DIR, RAW_DIR)
    
    # 归一化
    scaler = StandardScaler()
    N, T, F = data_matrix.shape # 注意这里的 shape 是 (Time, Nodes, Feat)
    # Reshape for scaling: (Time * Nodes, Feat)
    data_flat = data_matrix.reshape(-1, F)
    data_scaled = scaler.fit_transform(data_flat).reshape(N, T, F)
    
    X, Y = generate_dataset(data_scaled, CONFIG['seq_len'], CONFIG['pred_len'])
    
    print(f"[Train] Dataset shape: X {X.shape}, Y {Y.shape}")
    
    # 划分数据集
    split = int(len(X) * 0.8)
    X_train, X_val = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    Y_train, Y_val = torch.FloatTensor(Y[:split]), torch.FloatTensor(Y[split:])
    
    # 移动 adj 到设备
    adj = adj.to(CONFIG['device'])
    
    # DataLoader
    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 2. 初始化模型
    model = STGCN_Model(
        adj=adj, 
        input_dim=3, # dsi, order_rate, congestion
        hidden_dim=CONFIG['hidden_dim']
    ).to(CONFIG['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 3. 训练循环
    print("\n[Train] Starting training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(CONFIG['device']), by.to(CONFIG['device'])
            
            optimizer.zero_grad()
            out = model(bx) # (Batch, Nodes, 1)
            
            # by shape is (Batch, 1, Nodes). Squeeze it to (Batch, Nodes)
            target = by.squeeze(1) 
            pred = out.squeeze(2)
            
            # Masking: 只计算那些真实值不为0的节点的 Loss (避免拟合无效区域)
            mask = target != 0
            if mask.sum() > 0:
                loss = loss_fn(pred[mask], target[mask])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            bx, by = X_val.to(CONFIG['device']), Y_val.to(CONFIG['device'])
            out = model(bx)
            pred = out.squeeze(2).cpu().numpy()
            target = by.squeeze(1).cpu().numpy()
            
            # 简单评估 (Flatten)
            mae = mean_absolute_error(target.flatten(), pred.flatten())
            r2 = r2_score(target.flatten(), pred.flatten())
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Val MAE: {mae:.4f} | Val R2: {r2:.4f}")
            
    print("\n[Done] STGCN Training Finished.")
    torch.save(model.state_dict(), "outputs/stgcn_final.pt")

if __name__ == "__main__":
    main()