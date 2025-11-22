"""可视化28天不同时间段的压力区域热力图

从原始GeoJSON文件提取坐标，合并压力分数，按日期和时段生成热力图。
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("prediction_output")
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_ROUTES_DIR = Path("data/raw/ODIDMob_Routes")
STRESS_FILE = "xgb_optimized_stress.csv"

TIME_PERIODS = ["morning_peak", "lunch_peak", "day_off_peak", "dinner_peak", "night"]
TIME_PERIOD_NAMES = {
    "morning_peak": "早高峰",
    "lunch_peak": "午高峰", 
    "day_off_peak": "白天平峰",
    "dinner_peak": "晚高峰",
    "night": "夜间"
}


def extract_coordinates_from_geojson(file_path: Path) -> pd.DataFrame:
    """从GeoJSON文件提取路线坐标和基本信息"""
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get('features', [])
        for feat in features:
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})
            
            route_id = props.get('route_id') or props.get('Route_id') or props.get('id')
            if route_id is None:
                continue
            
            feature_type = props.get('feature_type', '')
            if feature_type != 'route':
                continue
            
            # 提取坐标
            gtype = geom.get('type', '')
            coordinates = geom.get('coordinates', [])
            
            if gtype == 'LineString' and len(coordinates) >= 2:
                # 计算路线中心点（所有点的平均）
                lons = [c[0] for c in coordinates if isinstance(c, list) and len(c) >= 2]
                lats = [c[1] for c in coordinates if isinstance(c, list) and len(c) >= 2]
                if lons and lats:
                    lat = float(np.mean(lats))
                    lon = float(np.mean(lons))
                    rows.append({
                        'route_id': str(route_id),
                        'lat': lat,
                        'lon': lon,
                        'date': props.get('date', '')
                    })
            elif gtype == 'Point' and isinstance(coordinates, list) and len(coordinates) >= 2:
                lat = float(coordinates[1])
                lon = float(coordinates[0])
                rows.append({
                    'route_id': str(route_id),
                    'lat': lat,
                    'lon': lon,
                    'date': props.get('date', '')
                })
    except Exception as e:
        print(f"[WARN] 读取 {file_path.name} 失败: {e}")
    
    return pd.DataFrame(rows)


def load_all_coordinates() -> pd.DataFrame:
    """加载所有日期的路线坐标"""
    all_dfs = []
    geojson_files = sorted(RAW_ROUTES_DIR.glob("DeliveryRoutes_*.geojson"))
    
    print(f"[INFO] 找到 {len(geojson_files)} 个GeoJSON文件")
    
    for fp in geojson_files:
        df = extract_coordinates_from_geojson(fp)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    result = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] 总共提取 {len(result)} 条路线坐标")
    return result


def merge_stress_scores(coords_df: pd.DataFrame) -> pd.DataFrame:
    """合并压力分数"""
    if not Path(STRESS_FILE).exists():
        print(f"[ERROR] 压力分数文件 {STRESS_FILE} 不存在")
        return pd.DataFrame()
    
    stress_df = pd.read_csv(STRESS_FILE, encoding='utf-8-sig')
    stress_df['route_id'] = stress_df['route_id'].astype(str)
    
    # 合并
    merged = coords_df.merge(
        stress_df[['route_id', 'date', 'time_period', 'optimized_stress_score']],
        on=['route_id', 'date'],
        how='inner'
    )
    
    print(f"[INFO] 合并后数据量: {len(merged)} 条记录")
    return merged


def create_grid_heatmap(df: pd.DataFrame, date: str, time_period: str, 
                        cell_size: float = 0.003) -> pd.DataFrame:
    """创建网格热力图数据
    
    Args:
        df: 包含 lat, lon, optimized_stress_score 的DataFrame
        date: 日期字符串 (YYYYMMDD)
        time_period: 时间段
        cell_size: 网格大小（度）
    
    Returns:
        网格聚合后的DataFrame
    """
    work = df[(df['date'].astype(str) == str(date)) & (df['time_period'] == time_period)].copy()
    
    if work.empty:
        return pd.DataFrame()
    
    # 创建网格
    work['grid_lat'] = (work['lat'] / cell_size).round() * cell_size
    work['grid_lon'] = (work['lon'] / cell_size).round() * cell_size
    work['grid_id'] = work['grid_lat'].astype(str) + '_' + work['grid_lon'].astype(str)
    
    # 聚合
    grid_agg = work.groupby('grid_id').agg({
        'optimized_stress_score': ['mean', 'count'],
        'grid_lat': 'first',
        'grid_lon': 'first'
    }).reset_index()
    
    grid_agg.columns = ['grid_id', 'avg_stress', 'count', 'lat', 'lon']
    grid_agg = grid_agg[grid_agg['count'] >= 1]  # 至少1个点
    
    return grid_agg


def plot_heatmap(grid_df: pd.DataFrame, date: str, time_period: str, 
                 save_path: Path):
    """绘制热力图"""
    if grid_df.empty:
        print(f"[WARN] {date} {time_period} 没有数据")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 散点图，颜色表示压力分数
    scatter = ax.scatter(
        grid_df['lon'],
        grid_df['lat'],
        c=grid_df['avg_stress'],
        s=grid_df['count'] * 20,  # 大小表示点数
        cmap='YlOrRd',
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5,
        vmin=0,
        vmax=12
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('平均压力分数', fontsize=12)
    
    # 设置标题和标签
    # 处理日期格式：可能是 YYYYMMDD 或 YYYY-MM-DD
    date_str = str(date)
    try:
        if '-' in date_str:
            date_str = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        else:
            date_str = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    except:
        date_str = str(date)  # 如果解析失败，使用原始字符串
    period_name = TIME_PERIOD_NAMES.get(time_period, time_period)
    ax.set_title(f'压力区域热力图 - {date_str} {period_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('经度', fontsize=11)
    ax.set_ylabel('纬度', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 保存: {save_path}")


def create_time_series_heatmap(all_data: pd.DataFrame, output_dir: Path):
    """创建时间序列热力图：显示28天×5个时段的压力变化"""
    
    # 按日期和时间段聚合
    time_series = all_data.groupby(['date', 'time_period'])['optimized_stress_score'].mean().reset_index()
    
    # Pivot为矩阵形式
    heatmap_data = time_series.pivot(index='time_period', columns='date', values='optimized_stress_score')
    
    # 确保时间段顺序
    time_order = [tp for tp in TIME_PERIODS if tp in heatmap_data.index]
    heatmap_data = heatmap_data.loc[time_order]
    
    # 重命名索引和列
    heatmap_data.index = [TIME_PERIOD_NAMES.get(tp, tp) for tp in heatmap_data.index]
    
    # 处理日期列格式
    formatted_cols = []
    for c in heatmap_data.columns:
        try:
            c_str = str(c)
            if '-' in c_str:
                date_obj = datetime.strptime(c_str, '%Y-%m-%d')
            else:
                date_obj = datetime.strptime(c_str, '%Y%m%d')
            formatted_cols.append(date_obj.strftime('%m-%d'))
        except:
            formatted_cols.append(str(c))
    heatmap_data.columns = formatted_cols
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=12)
    
    # 设置刻度
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    
    # 添加数值标注
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('平均压力分数', fontsize=12)
    
    ax.set_title('28天不同时段压力变化热力图', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('时间段', fontsize=11)
    
    plt.tight_layout()
    save_path = output_dir / 'stress_timeseries_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 保存时间序列热力图: {save_path}")


def main():
    print("[INFO] 开始生成压力区域热力图...")
    
    # 1. 加载坐标数据
    coords_df = load_all_coordinates()
    if coords_df.empty:
        print("[ERROR] 无法加载坐标数据")
        return
    
    # 2. 合并压力分数
    all_data = merge_stress_scores(coords_df)
    if all_data.empty:
        print("[ERROR] 无法合并压力分数")
        return
    
    print(f"[INFO] 数据日期范围: {all_data['date'].min()} 到 {all_data['date'].max()}")
    print(f"[INFO] 时间段: {all_data['time_period'].unique()}")
    
    # 3. 创建按日期和时段的热力图
    dates = sorted(all_data['date'].astype(str).unique())
    heatmap_dir = OUTPUT_DIR / 'heatmaps'
    heatmap_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] 生成 {len(dates)} 天 × {len(TIME_PERIODS)} 个时段的热力图...")
    
    count = 0
    for date in dates[:28]:  # 限制28天
        for period in TIME_PERIODS:
            grid_df = create_grid_heatmap(all_data, date, period)
            if not grid_df.empty:
                filename = f'heatmap_{date}_{period}.png'
                save_path = heatmap_dir / filename
                plot_heatmap(grid_df, date, period, save_path)
                count += 1
    
    print(f"[INFO] 总共生成 {count} 张热力图")
    
    # 4. 创建时间序列热力图
    print("[INFO] 生成时间序列热力图...")
    create_time_series_heatmap(all_data, OUTPUT_DIR)
    
    # 5. 生成汇总统计
    summary = all_data.groupby(['date', 'time_period'])['optimized_stress_score'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    summary_path = OUTPUT_DIR / 'heatmap_summary.csv'
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] 保存汇总统计: {summary_path}")
    
    print("[INFO] 热力图生成完成！")


if __name__ == '__main__':
    main()

