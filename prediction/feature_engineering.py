"""Feature engineering utilities for rider stress prediction.
Includes scaling, interaction features, simple kNN graph construction for STGCN.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def add_interactions(df: pd.DataFrame, pairs: List[Tuple[str,str]]) -> pd.DataFrame:
    work = df.copy()
    for a,b in pairs:
        if a in work.columns and b in work.columns:
            work[f"{a}__x__{b}"] = work[a] * work[b]
    return work

# --------------------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------------------

def build_similarity_graph(node_feature_matrix: np.ndarray, k: int = 5, threshold: float = 0.0) -> np.ndarray:
    """Build adjacency by cosine similarity + top-k selection.
    Returns binary adjacency (num_nodes, num_nodes).
    """
    if node_feature_matrix.size == 0:
        return np.zeros((0,0))
    sim = cosine_similarity(node_feature_matrix)
    np.fill_diagonal(sim, 0.0)
    # for each node pick top-k indices with similarity > threshold
    adj = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        scores = sim[i]
        idx = np.argsort(scores)[::-1]  # descending
        selected = [j for j in idx[:k] if scores[j] > threshold]
        adj[i, selected] = 1.0
    # make symmetric
    adj = np.maximum(adj, adj.T)
    return adj


def temporal_stack(pivot_df: pd.DataFrame, window: int, feature_cols: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Construct node feature tensors per time step for each courier.
    Returns list of 3D arrays: (window, num_nodes, num_features) and adjacency list per step shape (window, num_nodes, num_nodes).
    """
    sequences = []
    adjs = []
    for courier_id, g in pivot_df.groupby("courier_id"):
        g_sorted = g.sort_values("date")
        g_sorted = g_sorted.reset_index(drop=True)
        for idx in range(window, len(g_sorted)+1):
            hist = g_sorted.iloc[idx-window:idx]
            # node dimension = one node (single courier) for now -> can extend later to multi-rider graphs
            node_feats = hist[feature_cols].to_numpy(dtype=float)
            # shape (window, features); we reshape to (window,1,features)
            node_feats = node_feats.reshape(window,1,len(feature_cols))
            # adjacency trivial single node identity per time slice
            adj_window = np.ones((window,1,1), dtype=float)
            sequences.append(node_feats)
            adjs.append(adj_window)
    return sequences, adjs
