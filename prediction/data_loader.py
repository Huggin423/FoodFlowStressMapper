"""Data loading and sequence construction utilities for rider stress prediction.

This module loads daily processed rider feature CSVs, merges optional stress score outputs,
aggregates time_period level features, and constructs sliding window datasets for
traditional ML and (optionally) STGCN models.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# File discovery / loading
# --------------------------------------------------------------------------------------

def discover_processed_files(processed_root: str) -> List[Path]:
    root = Path(processed_root)
    files = sorted(root.glob("rider_features_*.csv"))
    return files


def load_all_routes(processed_root: str) -> pd.DataFrame:
    files = discover_processed_files(processed_root)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            # keep only route level for prediction stage
            route_df = df[df["feature_type"] == "route"].copy()
            dfs.append(route_df)
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

# --------------------------------------------------------------------------------------
# Optional stress score merges
# --------------------------------------------------------------------------------------

def merge_stress_scores(route_df: pd.DataFrame,
                        hierarchical_file: Optional[str] = None,
                        xgb_file: Optional[str] = None) -> pd.DataFrame:
    """Merge manual hierarchical stress scores and XGBoost-optimized scores (if provided)."""
    df = route_df.copy()

    # 先初始化两列，避免后面 KeyError
    df["stress_score_manual"] = np.nan
    df["stress_score_xgb"] = np.nan

    # hierarchical final（规则/层级版压力）
    if hierarchical_file and Path(hierarchical_file).exists():
        hdf = pd.read_csv(hierarchical_file, encoding="utf-8-sig")
        # 只取 route 级别
        mask_route = hdf["feature_type"] == "route"
        h_route = hdf.loc[mask_route, "route_id"].astype(str)
        h_scores = hdf.loc[mask_route, "stress_score"]
        h_map = dict(zip(h_route, h_scores))
        df["stress_score_manual"] = df["route_id"].astype(str).map(h_map)

    # xgb optimized（XGBoost 学到的优化压力）
    if xgb_file and Path(xgb_file).exists():
        xdf = pd.read_csv(xgb_file, encoding="utf-8-sig")
        # 这里列名是 optimized_stress_score，对应你的智慧城市2(1).py 输出
        x_map = dict(zip(xdf["route_id"].astype(str), xdf["optimized_stress_score"]))
        df["stress_score_xgb"] = df["route_id"].astype(str).map(x_map)

    # 统一目标列：优先 xgb，其次 manual
    df["stress_score"] = df["stress_score_xgb"].fillna(df["stress_score_manual"])

    return df

# --------------------------------------------------------------------------------------
# Aggregation to (courier_id, date, time_period)
# --------------------------------------------------------------------------------------

def aggregate_to_timeslice(route_df: pd.DataFrame) -> pd.DataFrame:
    if route_df.empty:
        return route_df
    # Basic numeric columns to aggregate (mean). Non-existing are ignored.
    candidate_cols = [
        "order_rate", "avg_interval", "continuous_orders", "load_intensity", "avg_speed", "spd_dev",
        "task_density", "nav_ratio", "task_per_km", "route_curvature", "weather_score",
        "congestion_index", "stress_score"
    ]
    avail = [c for c in candidate_cols if c in route_df.columns]
    agg_df = (
        route_df
        .groupby(["courier_id", "date", "time_period"], as_index=False)[avail]
        .mean()
    )
    # sorting by date for deterministic sequence
    agg_df.sort_values(["courier_id", "date"], inplace=True)
    return agg_df

# --------------------------------------------------------------------------------------
# Sliding window dataset builders
# --------------------------------------------------------------------------------------

TIME_PERIOD_ORDER = ["morning_peak", "lunch_peak", "day_off_peak", "dinner_peak", "night"]


def pivot_time_period(agg_df: pd.DataFrame,
                      enforce_all_periods: bool = True) -> pd.DataFrame:
    """Pivot time_period rows into wide columns per day per courier.
    Result index: (courier_id, date) columns: metric__time_period.
    """
    if agg_df.empty:
        return agg_df
    # ensure period column clean
    work = agg_df.copy()
    work["time_period"] = work["time_period"].fillna("unknown")
    pivot = (
        work
        .pivot_table(
            index=["courier_id", "date"],
            columns="time_period",
            values=[
                "order_rate", "avg_interval", "continuous_orders", "load_intensity",
                "avg_speed", "task_density", "task_per_km", "congestion_index",
                "stress_score"
            ],
            aggfunc="mean"
        )
    )
    # flatten multi index columns
    pivot.columns = [f"{metric}__{tp}" for metric, tp in pivot.columns]
    pivot.reset_index(inplace=True)

    if enforce_all_periods:
        # create missing columns for absent periods so model shape consistent
        base_metrics = [
            "order_rate", "avg_interval", "continuous_orders", "load_intensity",
            "avg_speed", "task_density", "task_per_km", "congestion_index",
            "stress_score"
        ]
        required = [f"{m}__{tp}" for m in base_metrics for tp in TIME_PERIOD_ORDER]
        for col in required:
            if col not in pivot.columns:
                pivot[col] = np.nan

    pivot.sort_values(["courier_id", "date"], inplace=True)
    return pivot


def add_lag_features(pivot_df: pd.DataFrame,
                     lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    df = pivot_df.copy()
    value_cols = [c for c in df.columns if c not in ["courier_id", "date"]]
    for lag in lags:
        lag_df = df.groupby("courier_id")[value_cols].shift(lag)
        lag_df.columns = [f"{c}_lag{lag}" for c in value_cols]
        df = pd.concat([df, lag_df], axis=1)
    return df


def build_supervised_samples(df: pd.DataFrame,
                             window: int = 7,
                             horizon: int = 1,
                             target_metric: str = "stress_score__day_off_peak"
                             ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Build (X,y) for ML models.
    X shape: (samples, features) after flattening window.
    Metadata list includes courier_id & target_date for each sample.
    
    Note: If a courier doesn't have enough history (window days), 
    we use available history with padding/zero-filling.
    """
    work = df.copy()
    # drop rows with insufficient data for target
    if target_metric not in work.columns:
        raise ValueError(f"Target metric {target_metric} not found in dataframe columns")

    samples_X = []
    samples_y = []
    meta: List[Dict] = []

    group = work.groupby("courier_id", sort=False)
    feat_cols = [c for c in work.columns if c not in ["courier_id", "date"]]
    feat_dim = len(feat_cols)
    
    for courier, g in group:
        g_sorted = g.sort_values("date").reset_index(drop=True)
        
        # 标准滑动窗口方式
        for idx in range(window, len(g_sorted) - horizon + 1):
            hist_slice = g_sorted.iloc[idx - window:idx]
            target_row = g_sorted.iloc[idx + horizon - 1]
            
            target_value = target_row[target_metric]
            if pd.isna(target_value):
                continue
            
            # flatten historical features
            hist_values = hist_slice[feat_cols].to_numpy().astype(float)
            # Handle NaN in historical features
            hist_values = np.nan_to_num(hist_values, nan=0.0, posinf=0.0, neginf=0.0)
            hist_values = hist_values.flatten()
            
            samples_X.append(hist_values)
            samples_y.append(target_value)
            meta.append({"courier_id": courier, "target_date": target_row["date"]})
        
        # 对于历史不足window天的骑手，使用可用历史（仅在至少1天历史时）
        if len(g_sorted) < window and len(g_sorted) > horizon:
            # 使用所有可用历史，不足部分用零填充
            for idx in range(len(g_sorted) - horizon, len(g_sorted)):
                if idx < 1:  # 至少需要1天历史
                    continue
                hist_slice = g_sorted.iloc[:idx]  # 使用所有可用历史
                target_row = g_sorted.iloc[idx + horizon - 1] if idx + horizon - 1 < len(g_sorted) else g_sorted.iloc[-1]
                
                target_value = target_row[target_metric]
                if pd.isna(target_value):
                    continue
                
                # 构建历史特征向量
                hist_values = hist_slice[feat_cols].to_numpy().astype(float)
                hist_values = np.nan_to_num(hist_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 不足window天的部分用零填充
                if len(hist_values) < window:
                    padding = np.zeros((window - len(hist_values), feat_dim))
                    hist_values = np.vstack([padding, hist_values])
                
                hist_values = hist_values.flatten()
                samples_X.append(hist_values)
                samples_y.append(target_value)
                meta.append({"courier_id": courier, "target_date": target_row["date"]})

    if not samples_X:
        print(f"[WARN] No supervised samples generated. Window={window}, horizon={horizon}")
        print(f"[WARN] Total groups processed: {len(list(group))}")
        return np.empty((0, 0)), np.empty((0,)), []
    X = np.vstack(samples_X)
    y = np.array(samples_y, dtype=float)
    print(f"[INFO] Generated {len(samples_X)} supervised samples from {len(list(group))} couriers")
    return X, y, meta

# --------------------------------------------------------------------------------------
# Utility to unify full pipeline
# --------------------------------------------------------------------------------------

def prepare_dataset(processed_root: str,
                    hierarchical_file: Optional[str] = None,
                    xgb_file: Optional[str] = None,
                    lags: List[int] = [1, 2, 3],
                    window: int = 7,
                    horizon: int = 1,
                    target_metric: str = "stress_score__day_off_peak"):
    """Full pipeline:
    1) load all route-level features from 28-day preprocessed CSVs
    2) merge hierarchical & xgb stress scores
    3) aggregate to (courier_id, date, time_period)
    4) pivot to wide format and add temporal lags
    5) build sliding-window supervised samples
    """
    route_df = load_all_routes(processed_root)
    if route_df.empty:
        print("[ERROR] No route data loaded.")
        return None

    merged = merge_stress_scores(route_df, hierarchical_file, xgb_file)

    # ⭐ 关键修复：统一 date 类型，避免 sort_values("date") 时出现 str/int 混合比较报错
    merged = merged.copy()
    merged["date"] = pd.to_datetime(merged["date"].astype(str), errors="coerce")
    # 丢掉无法解析日期的记录（通常是极少数异常）
    merged = merged.dropna(subset=["date"])

    agg = aggregate_to_timeslice(merged)
    pivot = pivot_time_period(agg)
    pivot = add_lag_features(pivot, lags=lags)
    X, y, meta = build_supervised_samples(pivot,
                                          window=window,
                                          horizon=horizon,
                                          target_metric=target_metric)
    return {
        "raw_routes": route_df,
        "merged": merged,
        "agg_timeslice": agg,
        "pivot": pivot,
        "X": X,
        "y": y,
        "meta": meta
    }


if __name__ == "__main__":
    # Example usage (paths adapted to project structure)
    processed_dir = os.path.join("data", "processed", "preprocessed_data(28_days)")
    # Provide optional stress score files if already generated
    hierarchical = "rider_stress_hierarchical_final.csv"  # adjust path if needed
    xgb_opt = "xgb_optimized_stress.csv"                  # adjust path if needed
    ds = prepare_dataset(processed_dir,
                         hierarchical_file=hierarchical,
                         xgb_file=xgb_opt)
    if ds:
        print("Loaded routes:", len(ds["raw_routes"]))
        print("Pivot shape:", ds["pivot"].shape)
        print("Supervised samples X,y:", ds["X"].shape, ds["y"].shape)
