"""Spatial hotspot clustering using a simplified ST-DBSCAN on route centroids.

Assumptions:
- Raw route geometry JSON/GeoJSON files located in data/raw/map (each file may contain a single
  route feature or feature collection). We attempt to parse 'features' list or treat top-level as a feature.
- Each route has an identifier route_id which must align with stress score files (hierarchical / xgb optimized).
- If geometry missing or parsing fails, that route is skipped.

Outputs:
- prediction_output/hotspot_points.csv : per route centroid with stress & cluster label
- prediction_output/hotspot.geojson : aggregated grid cells with average stress & cluster majority
- prediction_output/hotspot_cluster_summary.csv : cluster stats

Parameters can be tuned in main().
"""
from __future__ import annotations
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

OUTPUT_DIR = Path('prediction_output')
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_MAP_DIR = Path('data/raw/map')  # adjust if needed

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def _safe_read_json(p: Path) -> Any:
    try:
        with open(p,'r',encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def extract_route_records() -> pd.DataFrame:
    rows = []
    for fp in RAW_MAP_DIR.glob('*.json'):
        data = _safe_read_json(fp)
        if not data:
            continue
        features = []
        if isinstance(data, dict):
            if 'features' in data and isinstance(data['features'], list):
                features = data['features']
            else:
                # maybe single feature encoded directly
                if 'geometry' in data and 'properties' in data:
                    features = [data]
        for feat in features:
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})
            route_id = props.get('route_id') or props.get('Route_id') or props.get('id')
            feature_type = props.get('feature_type','')
            if feature_type != 'route':
                continue
            coords = []
            gtype = geom.get('type')
            if gtype == 'LineString':
                coords = geom.get('coordinates', [])
            elif gtype == 'Point':
                c = geom.get('coordinates')
                if isinstance(c, list):
                    coords = [c]
            if not coords:
                continue
            # centroid of linestring/points
            lat_list = [c[1] for c in coords if isinstance(c,list) and len(c)>=2]
            lon_list = [c[0] for c in coords if isinstance(c,list) and len(c)>=2]
            if not lat_list or not lon_list:
                continue
            lat = float(sum(lat_list)/len(lat_list))
            lon = float(sum(lon_list)/len(lon_list))
            date = props.get('date')
            time_period = props.get('time_period')
            rows.append({'route_id': str(route_id), 'lat': lat, 'lon': lon, 'date': date, 'time_period': time_period})
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------------------
# Stress score merge
# --------------------------------------------------------------------------------------

def merge_stress(route_df: pd.DataFrame, hierarchical_file: str = 'rider_stress_hierarchical_final.csv',
                 xgb_file: str = 'xgb_optimized_stress.csv') -> pd.DataFrame:
    df = route_df.copy()
    score_col = None
    if Path(xgb_file).exists():
        xdf = pd.read_csv(xgb_file, encoding='utf-8-sig')
        xmap = dict(zip(xdf['route_id'].astype(str), xdf['optimized_stress_score']))
        df['stress_score'] = df['route_id'].map(xmap)
        score_col = 'optimized_stress_score'
    if df['stress_score'].isna().all() and Path(hierarchical_file).exists():
        hdf = pd.read_csv(hierarchical_file, encoding='utf-8-sig')
        hroute = hdf[hdf['feature_type']=='route']
        hmap = dict(zip(hroute['route_id'].astype(str), hroute['stress_score']))
        df['stress_score'] = df['route_id'].map(hmap)
        score_col = 'stress_score'
    return df

# --------------------------------------------------------------------------------------
# Grid aggregation
# --------------------------------------------------------------------------------------

def assign_grid(df: pd.DataFrame, cell_size_deg: float = 0.003) -> pd.DataFrame:
    work = df.copy()
    work['grid_lat'] = (work['lat']/cell_size_deg).round()*cell_size_deg
    work['grid_lon'] = (work['lon']/cell_size_deg).round()*cell_size_deg
    work['grid_id'] = work['grid_lat'].astype(str) + '_' + work['grid_lon'].astype(str)
    return work

# --------------------------------------------------------------------------------------
# ST-DBSCAN simplified implementation
# --------------------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c


def time_encode(row) -> float:
    # Map time_period to nominal hour center
    mapping = {
        'morning_peak': 8,
        'lunch_peak': 12,
        'day_off_peak': 15,
        'dinner_peak': 18,
        'night': 23
    }
    th = mapping.get(row.get('time_period'), 12)
    date = str(row.get('date'))
    # simplistic date to day index (YYYYMMDD)
    if len(date)==8 and date.isdigit():
        y = int(date[0:4]); m = int(date[4:6]); d = int(date[6:8])
        # convert to ordinal
        import datetime
        ordinal = datetime.date(y,m,d).toordinal()
    else:
        ordinal = 0
    return ordinal*24 + th


def st_dbscan(df: pd.DataFrame, eps_space_m: float = 300.0, eps_time_h: float = 2.0, min_samples: int = 5) -> pd.DataFrame:
    if df.empty:
        df['cluster'] = []
        return df
    work = df.copy()
    work['t_val'] = work.apply(time_encode, axis=1)
    coords = work[['lat','lon','t_val']].to_numpy()
    n = coords.shape[0]
    visited = np.zeros(n, dtype=bool)
    cluster_labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    def region_query(idx):
        neighbors = []
        (lat1,lon1,t1) = coords[idx]
        for j in range(n):
            if j==idx: continue
            lat2,lon2,t2 = coords[j]
            spatial = haversine(lat1,lon1,lat2,lon2)
            temporal = abs(t2 - t1)/24.0  # convert back to hours diff approx
            if spatial <= eps_space_m and temporal <= eps_time_h:
                neighbors.append(j)
        return neighbors

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            cluster_labels[i] = -1
        else:
            cluster_labels[i] = cluster_id
            seeds = neighbors.copy()
            k = 0
            while k < len(seeds):
                s = seeds[k]
                if not visited[s]:
                    visited[s] = True
                    s_neighbors = region_query(s)
                    if len(s_neighbors) >= min_samples:
                        # append new
                        for nb in s_neighbors:
                            if nb not in seeds:
                                seeds.append(nb)
                if cluster_labels[s] == -1:
                    cluster_labels[s] = cluster_id
                elif cluster_labels[s] == -1:
                    cluster_labels[s] = cluster_id
                k += 1
            cluster_id += 1
    work['cluster'] = cluster_labels
    return work

# --------------------------------------------------------------------------------------
# GeoJSON output
# --------------------------------------------------------------------------------------

def export_geojson(grid_df: pd.DataFrame, path: Path):
    features = []
    for _,row in grid_df.iterrows():
        lat = row['grid_lat']; lon = row['grid_lon']
        # approximate square polygon (not real geo accuracy, but placeholder)
        size = 0.0015  # half cell in degrees
        polygon = [
            [lon-size, lat-size], [lon+size, lat-size], [lon+size, lat+size], [lon-size, lat+size], [lon-size, lat-size]
        ]
        feat = {
            'type': 'Feature',
            'geometry': {'type': 'Polygon','coordinates':[polygon]},
            'properties': {
                'grid_id': row['grid_id'],
                'avg_stress': row['avg_stress'],
                'count': int(row['count']),
                'cluster_majority': int(row['cluster_majority'])
            }
        }
        features.append(feat)
    geo = {'type':'FeatureCollection','features':features}
    with open(path,'w',encoding='utf-8') as f:
        json.dump(geo,f,ensure_ascii=False,indent=2)

# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main():
    routes_df = extract_route_records()
    if routes_df.empty:
        print('[ERROR] 未从 raw/map 解析到路线数据。请检查原始文件结构。')
        return
    routes_df = merge_stress(routes_df)
    routes_df = routes_df.dropna(subset=['stress_score'])
    if routes_df.empty:
        print('[ERROR] 路线压力分数全部缺失。确保已生成 xgb 或 hierarchical 压力文件。')
        return
    clustered = st_dbscan(routes_df)
    clustered.to_csv(OUTPUT_DIR / 'hotspot_points.csv', index=False, encoding='utf-8-sig')
    print('[INFO] 保存 hotspot_points.csv')
    grid_df = assign_grid(clustered)
    agg = (grid_df.groupby('grid_id')
           .agg(avg_stress=('stress_score','mean'),
                 count=('grid_id','count'),
                 grid_lat=('grid_lat','first'),
                 grid_lon=('grid_lon','first'),
                 cluster_majority=('cluster', lambda x: x.value_counts().idxmax()))
           .reset_index())
    agg.to_csv(OUTPUT_DIR / 'hotspot_cluster_summary.csv', index=False, encoding='utf-8-sig')
    print('[INFO] 保存 hotspot_cluster_summary.csv')
    export_geojson(agg, OUTPUT_DIR / 'hotspot.geojson')
    print('[INFO] 保存 hotspot.geojson')

if __name__ == '__main__':
    main()
