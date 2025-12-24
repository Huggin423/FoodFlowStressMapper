"""
时空图构建模块 - Spatio-Temporal Graph Construction
用于STGCN模型的图结构构建
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
import math


class SpatialGraphBuilder:
    """空间图构建器"""
    
    def __init__(self, method='grid', grid_size=500):
        """
        Args:
            method: 'grid' - 基于网格邻接, 'knn' - K近邻, 'similarity' - 特征相似度
            grid_size: 网格大小(米)
        """
        self.method = method
        self.grid_size = grid_size
    
    def build_grid_graph(self, data_with_coords):
        """
        基于网格划分构建空间邻接图
        
        Args:
            data_with_coords: DataFrame,需包含'lon'和'lat'列
        
        Returns:
            adj_matrix: (N, N) 邻接矩阵
            grid_ids: 每个样本的网格ID
        """
        if 'lon' not in data_with_coords.columns or 'lat' not in data_with_coords.columns:
            raise ValueError("Data must contain 'lon' and 'lat' columns")
        
        # 将经纬度映射到网格
        lons = data_with_coords['lon'].values
        lats = data_with_coords['lat'].values
        
        # 计算网格ID
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        # 经纬度1度约111km,转换为网格数
        lon_cells = int((lon_max - lon_min) * 111000 / self.grid_size) + 1
        lat_cells = int((lat_max - lat_min) * 111000 / self.grid_size) + 1
        
        grid_ids = []
        for lon, lat in zip(lons, lats):
            col = int((lon - lon_min) / (lon_max - lon_min) * lon_cells)
            row = int((lat - lat_min) / (lat_max - lat_min) * lat_cells)
            grid_id = row * lon_cells + col
            grid_ids.append(grid_id)
        
        # 构建邻接矩阵:相邻网格相连
        n = len(data_with_coords)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # 计算网格距离
                grid_i, grid_j = grid_ids[i], grid_ids[j]
                row_i, col_i = grid_i // lon_cells, grid_i % lon_cells
                row_j, col_j = grid_j // lon_cells, grid_j % lon_cells
                
                # 曼哈顿距离<=1则为邻居
                if abs(row_i - row_j) + abs(col_i - col_j) <= 1:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # 添加自环
        np.fill_diagonal(adj_matrix, 1)
        
        return adj_matrix, grid_ids
    
    def build_knn_graph(self, data_with_coords, k=5):
        """
        基于K近邻构建空间图
        
        Args:
            data_with_coords: DataFrame,需包含'lon'和'lat'列
            k: 邻居数量
        
        Returns:
            adj_matrix: (N, N) 邻接矩阵
        """
        coords = data_with_coords[['lon', 'lat']].values
        
        # 使用KDTree查找K近邻
        tree = cKDTree(coords)
        n = len(coords)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            distances, indices = tree.query(coords[i], k=k+1)  # +1因为包含自己
            for j in indices[1:]:  # 跳过自己
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # 添加自环
        np.fill_diagonal(adj_matrix, 1)
        
        return adj_matrix
    
    def build_similarity_graph(self, features, threshold=0.5):
        """
        基于特征相似度构建图
        
        Args:
            features: (N, F) 特征矩阵
            threshold: 相似度阈值
        
        Returns:
            adj_matrix: (N, N) 邻接矩阵
        """
        # 计算余弦相似度
        sim_matrix = cosine_similarity(features)
        
        # 阈值化
        adj_matrix = (sim_matrix >= threshold).astype(float)
        
        # 确保自环
        np.fill_diagonal(adj_matrix, 1)
        
        return adj_matrix
    
    def normalize_adjacency(self, adj_matrix):
        """
        归一化邻接矩阵 (对称归一化)
        D^{-1/2} A D^{-1/2}
        """
        # 计算度矩阵
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        # 归一化
        adj_normalized = d_mat_inv_sqrt @ adj_matrix @ d_mat_inv_sqrt
        
        return adj_normalized


class TemporalFeatureBuilder:
    """时序特征构建器"""
    
    @staticmethod
    def add_time_features(data):
        """
        添加时间相关特征
        
        Args:
            data: DataFrame,需包含'date'和'time_period'列
        
        Returns:
            data: 添加了时间特征的DataFrame
        """
        data = data.copy()
        
        # 时段编码
        time_period_map = {
            'morning_peak': 0,
            'day_off_peak': 1,
            'lunch_peak': 2,
            'dinner_peak': 3,
            'night': 4
        }
        data['time_period_code'] = data['time_period'].map(time_period_map).fillna(1)
        
        # 日期特征
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', errors='coerce')
            data['day_of_week'] = data['date'].dt.dayofweek  # 0=周一
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            data['day_of_month'] = data['date'].dt.day
        
        return data
    
    @staticmethod
    def create_lag_features(data, group_cols, target_col, lags=[1, 2, 3]):
        """
        创建滞后特征
        
        Args:
            data: DataFrame
            group_cols: 分组列(如['courier_id'])
            target_col: 目标列(如'dsi')
            lags: 滞后期数
        
        Returns:
            data: 添加了滞后特征的DataFrame
        """
        data = data.copy()
        data = data.sort_values(group_cols + ['date'])
        
        for lag in lags:
            col_name = f'{target_col}_lag{lag}'
            data[col_name] = data.groupby(group_cols)[target_col].shift(lag)
        
        return data


class HeterogeneousGraphBuilder:
    """异构图构建器(骑手-区域二部图)"""
    
    def __init__(self):
        pass
    
    def build_bipartite_graph(self, courier_data, grid_ids):
        """
        构建骑手-网格二部图
        
        Args:
            courier_data: DataFrame,包含courier_id
            grid_ids: 每个样本对应的网格ID
        
        Returns:
            adj_matrix: (N, N) 异构邻接矩阵
        """
        n = len(courier_data)
        adj_matrix = np.zeros((n, n))
        
        courier_ids = courier_data['courier_id'].values
        
        # 同一骑手在不同时空的记录相连
        for i in range(n):
            for j in range(i+1, n):
                # 同一骑手
                if courier_ids[i] == courier_ids[j]:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                # 同一网格
                elif grid_ids[i] == grid_ids[j]:
                    adj_matrix[i, j] = 0.5  # 权重降低
                    adj_matrix[j, i] = 0.5
        
        np.fill_diagonal(adj_matrix, 1)
        
        return adj_matrix


def aggregate_to_grid(data, grid_size=500):
    """
    将点数据聚合到网格
    
    Args:
        data: DataFrame,包含'lon','lat'和目标变量
        grid_size: 网格大小(米)
    
    Returns:
        grid_data: 聚合后的网格数据
    """
    if 'lon' not in data.columns or 'lat' not in data.columns:
        raise ValueError("Data must contain 'lon' and 'lat' columns")
    
    # 计算网格ID
    lons = data['lon'].values
    lats = data['lat'].values
    
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    
    lon_cells = int((lon_max - lon_min) * 111000 / grid_size) + 1
    lat_cells = int((lat_max - lat_min) * 111000 / grid_size) + 1
    
    grid_ids = []
    for lon, lat in zip(lons, lats):
        col = int((lon - lon_min) / (lon_max - lon_min) * lon_cells)
        row = int((lat - lat_min) / (lat_max - lat_min) * lat_cells)
        grid_id = row * lon_cells + col
        grid_ids.append(grid_id)
    
    data['grid_id'] = grid_ids
    
    # 按网格聚合
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    agg_dict = {col: 'mean' for col in numeric_cols if col not in ['grid_id', 'courier_id']}
    
    grid_data = data.groupby('grid_id').agg(agg_dict).reset_index()
    
    # 计算网格中心坐标
    grid_data['grid_lon'] = (grid_data['grid_id'] % lon_cells + 0.5) / lon_cells * (lon_max - lon_min) + lon_min
    grid_data['grid_lat'] = (grid_data['grid_id'] // lon_cells + 0.5) / lat_cells * (lat_max - lat_min) + lat_min
    
    return grid_data


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例:构建空间图
    print("时空图构建模块已加载")
    print("可用的类:")
    print("  - SpatialGraphBuilder: 空间图构建")
    print("  - TemporalFeatureBuilder: 时序特征")
    print("  - HeterogeneousGraphBuilder: 异构图构建")
    print("\n使用示例:")
    print("  builder = SpatialGraphBuilder(method='grid', grid_size=500)")
    print("  adj_matrix, grid_ids = builder.build_grid_graph(data)")
