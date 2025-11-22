"""Training script for rider stress prediction.
Supports traditional ML regression (XGBoost) and optional STGCN deep model.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Attempt XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Attempt PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from models.stgcn import build_stgcn
    PYTORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    PYTORCH_AVAILABLE = False

from data_loader import prepare_dataset

OUTPUT_DIR = Path("prediction_output")
OUTPUT_DIR.mkdir(exist_ok=True)


class STWindowDataset(Dataset):
    def __init__(self, pivot_df: pd.DataFrame, window: int, feature_cols, target_metric):
        self.samples = []
        self.adjs = []
        self.targets = []
        for courier, g in pivot_df.groupby("courier_id"):
            g = g.sort_values("date").reset_index(drop=True)
            for idx in range(window, len(g)):
                hist = g.iloc[idx - window:idx]
                cur = g.iloc[idx]
                if (target_metric not in cur) or pd.isna(cur[target_metric]):
                    continue
                feats = hist[feature_cols].to_numpy(dtype=float)  # (window, features)
                feats = feats.reshape(window, 1, len(feature_cols))
                adj = np.ones((1, 1), dtype=float)  # single node graph
                self.samples.append(feats)
                self.adjs.append(adj)
                self.targets.append(cur[target_metric])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.adjs[idx], self.targets[idx])


def train_xgb(X: np.ndarray, y: np.ndarray) -> Dict:
    if xgb is None:
        raise RuntimeError("xgboost not installed.")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
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
    preds = model.predict(X_val)
    metrics = {
        "MAE": float(mean_absolute_error(y_val, preds)),
        "RMSE": float(mean_squared_error(y_val, preds) ** 0.5),
        "R2": float(r2_score(y_val, preds)),
    }
    model_path = OUTPUT_DIR / "xgb_model.json"
    model.save_model(model_path)
    return {"model": model, "metrics": metrics, "model_path": str(model_path)}


def train_stgcn(pivot_df: pd.DataFrame, window: int, target_metric: str):
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    # 使用除 stress_score__* 以外的特征列
    feature_cols = [
        c
        for c in pivot_df.columns
        if c not in ["courier_id", "date"] and not c.startswith("stress_score__")
    ]
    ds = STWindowDataset(
        pivot_df, window=window, feature_cols=feature_cols, target_metric=target_metric
    )
    if len(ds) == 0:
        raise RuntimeError("No STGCN samples available.")
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = build_stgcn(
        in_features=len(feature_cols), hidden=32, out_features=1, blocks=2
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(10):  # small epochs for demo
        model.train()
        epoch_loss = 0.0
        for feats, adj, tgt in loader:
            # Convert numpy arrays to torch tensors
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats).float()
            else:
                feats = torch.tensor(feats, dtype=torch.float32)
            
            if isinstance(adj, np.ndarray):
                adj = torch.from_numpy(adj).float()
            else:
                adj = torch.tensor(adj, dtype=torch.float32)
            
            if isinstance(tgt, np.ndarray):
                tgt = torch.from_numpy(tgt).float()
            else:
                tgt = torch.tensor(tgt, dtype=torch.float32)
            
            # feats: (batch, window, 1, features)
            batch_size = feats.size(0)
            # adj should be (batch, nodes, nodes) = (batch, 1, 1)
            # Each sample has adj shape (1, 1), need to replicate for batch
            if adj.dim() == 2:  # single sample (1, 1)
                adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
            elif adj.size(0) != batch_size:
                # If batch dimension doesn't match, take first sample and replicate
                adj = adj[0:1].repeat(batch_size, 1, 1)
            
            optim.zero_grad()
            preds = model(feats, adj)  # (batch, 1)
            preds = preds.view(-1)
            loss = loss_fn(preds, tgt)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * batch_size
        print(f"[STGCN] Epoch {epoch+1} loss {epoch_loss/len(ds):.4f}")
    model_path = OUTPUT_DIR / "stgcn.pt"
    torch.save(model.state_dict(), model_path)
    return {"model": model, "model_path": str(model_path)}


def main():
    processed_dir = os.path.join("data", "processed", "preprocessed_data(28_days)")
    hierarchical = "rider_stress_hierarchical_final.csv"
    xgb_opt = "xgb_optimized_stress.csv"

    print("[INFO] Loading data...")
    # 使用较小的窗口以适应实际数据情况（平均每个骑手只有2-3天数据）
    ds = prepare_dataset(
        processed_dir,
        hierarchical_file=hierarchical,
        xgb_file=xgb_opt,
        window=3,  # 降低窗口大小到3天
        horizon=1,
        target_metric="stress_score__day_off_peak",
    )
    if ds is None:
        print("[ERROR] Failed to load dataset.")
        return

    pivot_df = ds["pivot"]
    print(f"[INFO] Loaded pivot dataframe: shape={pivot_df.shape}")
    print(f"[INFO] Date range: {pivot_df['date'].min()} to {pivot_df['date'].max()}")
    print(f"[INFO] Unique couriers: {pivot_df['courier_id'].nunique()}")
    print(f"[INFO] Average days per courier: {len(pivot_df) / pivot_df['courier_id'].nunique():.2f}")
    
    report: Dict = {}

    # -------------------- 1) 基线 XGBoost：按天样本回归 --------------------
    # 尝试多个目标时段，选择有数据的
    target_metrics = [
        "stress_score__day_off_peak",
        "stress_score__lunch_peak",
        "stress_score__dinner_peak",
        "stress_score__morning_peak",
        "stress_score__night"
    ]
    
    target_metric = None
    for tm in target_metrics:
        if tm in pivot_df.columns and pivot_df[tm].notna().sum() > 100:
            target_metric = tm
            print(f"[INFO] Selected target metric: {target_metric}")
            print(f"[INFO] Valid samples: {pivot_df[target_metric].notna().sum()}")
            break
    
    if target_metric is None:
        print("[WARN] No suitable target metric found. Trying first available...")
        stress_cols = [c for c in pivot_df.columns if c.startswith("stress_score__") and not c.endswith("_lag")]
        if stress_cols:
            target_metric = stress_cols[0]
            print(f"[INFO] Using: {target_metric}")
        else:
            print("[ERROR] No stress score columns found!")
            return

    if target_metric not in pivot_df.columns:
        print(f"[ERROR] Target metric {target_metric} not in pivot dataframe.")
        return

    # 特征：所有非 stress_score__* 的列，但保留lag特征
    feature_cols = [
        c
        for c in pivot_df.columns
        if c not in ["courier_id", "date"]
        and not (c.startswith("stress_score__") and not c.endswith("_lag"))
    ]
    
    # 处理NaN值
    mask = pivot_df[target_metric].notna()
    X_full = pivot_df.loc[mask, feature_cols].copy()
    y_full = pivot_df.loc[mask, target_metric].copy()
    
    # 填充特征中的NaN值
    X_full = X_full.fillna(X_full.median())
    
    X = X_full.to_numpy(dtype=float)
    y = y_full.to_numpy(dtype=float)
    
    print(f"[INFO] Final dataset: X={X.shape}, y={y.shape}")
    print(f"[INFO] Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

    if X.size > 0:
        if xgb is None:
            print("[WARN] XGBoost not installed. Install it with: pip install xgboost")
            print("[INFO] Skipping XGBoost training. Dataset ready for training when XGBoost is installed.")
            print(f"[INFO] Dataset stats: X={X.shape}, y={y.shape}, target_range=[{y.min():.2f}, {y.max():.2f}]")
        else:
            print("[INFO] Training XGBoost baseline (daily-level regression)...")
            try:
                report["xgb"] = train_xgb(X, y)
                print("[INFO] XGBoost metrics:", report["xgb"]["metrics"])
                report["xgb"]["target_metric"] = target_metric
                report["xgb"]["feature_count"] = len(feature_cols)
                report["xgb"]["sample_count"] = len(X)
            except Exception as e:
                print(f"[ERROR] XGBoost training failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("[WARN] Skipping XGBoost (no valid data).")

    # -------------------- 2) STGCN（窗口序列建模，demo 用） --------------------
    try:
        if torch is not None and target_metric:
            print("[INFO] Training STGCN demo model...")
            # 使用更短窗口，提高有样本的概率
            stgcn_info = train_stgcn(
                pivot_df, window=2, target_metric=target_metric  # 进一步降低窗口到2天
            )
            report["stgcn"] = {"model_path": stgcn_info["model_path"]}
            report["stgcn"]["target_metric"] = target_metric
        else:
            print("[WARN] PyTorch not installed or no target metric, skipping STGCN.")
    except Exception as e:
        print(f"[WARN] STGCN training failed: {e}")
        import traceback
        traceback.print_exc()

    # -------------------- 3) 保存训练报告 --------------------
    report["dataset_info"] = {
        "total_samples": len(pivot_df),
        "unique_couriers": pivot_df['courier_id'].nunique(),
        "date_range": [str(pivot_df['date'].min()), str(pivot_df['date'].max())],
        "avg_days_per_courier": len(pivot_df) / pivot_df['courier_id'].nunique()
    }
    
    metrics_path = OUTPUT_DIR / "training_report.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] Saved training report to {metrics_path}")


if __name__ == "__main__":
    main()
