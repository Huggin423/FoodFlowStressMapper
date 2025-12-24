# AI Coding Agent Instructions - Rider Stress Prediction

## 🎯 Project Overview
A **research framework for delivery rider stress prediction** using physics-based metrics (DSI - Delivery Stress Index) with spatio-temporal modeling. This is an academic project comparing baseline (XGBoost) vs. proposed (ConvLSTM/STGCN) models.

**Key Research Focus:** Prevent data leakage, use time-lagged features, model actual spatio-temporal dynamics.

---

## 🏗️ Architecture & Data Flow

### Core Components

1. **Data Preprocessing** (`process/data_preprocessor.py`)
   - Input: GeoJSON files (`DeliveryRoutes_YYYYMMDD.geojson`) with rider trajectories
   - Process: Extract coordinates, timestamps, compute physics-based stress components
   - Output: `output_features/rider_features_YYYYMMDD.csv` with DSI column
   - **Critical:** Handles date parsing from filenames, JSON field extraction, entropy calculations

2. **Feature Engineering** (`prediction/train.py` - lines 90-190)
   - **Time-Lag Features:** Create lagged versions of features for T-1, T-2, etc.
   - **Remove Leakage:** Exclude `speed_strain` and `load_strain` from X (these ARE the DSI components)
   - **Target:** Predict DSI at T+1 using features from T and earlier
   - Function: `create_time_lag_features()` groups by courier_id, builds sliding windows

3. **Model Training** (`prediction/train.py`)
   - **Baseline:** XGBoost on tabular time-lagged features (lines 400-460)
   - **Proposed:** ConvLSTM for actual spatio-temporal processing (lines 220-390)
   - **Evaluation:** Compare validation R² to measure improvement

### Data Flow
```
GeoJSON files → Preprocessor (DSI calc) → output_features/ CSV
         ↓
    load_preprocessed_data() → create_time_lag_features() 
         ↓
prepare_features_and_target_no_leakage() → Train/Val split
         ↓
XGBoost Baseline │ ConvLSTM Proposed → Compare R²
```

---

## 🔑 Critical Patterns & Conventions

### 1. **Anti-Data-Leakage Pattern**
```python
# ❌ WRONG: Using speed_strain/load_strain in X (they define DSI!)
X = data[['speed_strain', 'load_strain', 'order_rate', ...]]
y = data['dsi']  # Data leakage!

# ✓ RIGHT: Time-lagged features without leakage components
X = data[['lag_1_dsi', 'lag_2_dsi', 'order_rate', 'avg_speed', ...]]
y = data['dsi_target']  # T+1 prediction
```
- Always use `create_time_lag_features()` before modeling
- Verify `speed_strain` and `load_strain` are NOT in feature list
- Target should be `dsi_target` (future DSI), not current `dsi`

### 2. **Grouping by Courier ID**
Data must be grouped by `courier_id` before temporal operations:
```python
data_sorted = data.sort_values(['courier_id', 'date']).reset_index(drop=True)
for courier_id, group_df in data_sorted.groupby('courier_id'):
    # Create lag/lead features per courier sequence
```
Don't cross sequences between different couriers.

### 3. **DSI Components**
DSI = f(speed_strain, load_strain), where:
- `speed_strain`: deviation from normal speed
- `load_strain`: deviation from normal load
- **Never** use these directly as input features when predicting DSI
- Instead: use past DSI values + other non-derived features

### 4. **Output Directory Structure**
```
outputs/
├── xgboost_baseline.json       # Model file
├── convlstm_best.pt            # PyTorch checkpoint
└── experiment_report.json       # Results summary
```
Always save models to `outputs/` (created automatically).

---

## 🚀 Essential Workflows

### Running Full Pipeline
```bash
# 1. Preprocess data (generates output_features/)
cd process
python data_preprocessor.py

# 2. Train models with comparison
cd ../prediction
python train.py
```

### Key Functions to Know
- `load_preprocessed_data(data_dir)` - Load CSV files from a directory
- `create_time_lag_features(data, lag_steps=3)` - Create T-1, T-2, T-3 features
- `prepare_features_and_target_no_leakage()` - Final X, y preparation
- `train_xgboost_baseline()` - Baseline model training
- `train_convlstm_model()` - ConvLSTM (requires PyTorch)

### Checking Model Performance
Validation R² should be realistic (0.3-0.7 range), not suspiciously high (0.99). High R² suggests data leakage.

---

## 📦 Dependencies

### Core (Always)
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tqdm  # Progress bars
```

### Optional (for deep learning)
```
torch>=1.10.0  # ConvLSTM model
```

### For Preprocessing
```
geopandas>=0.10.0  # GeoJSON parsing
```

---

## 🔗 Integration Points

1. **GeoJSON Files** - Input data format for preprocessor
   - Required columns: route geometry, activity list, timestamps
   - Location: `data/raw/ODIDMob_Routes/`

2. **Graph Construction** (`prediction/graph_builder.py`)
   - Used for potential STGCN implementation
   - Methods: grid-based, KNN-based, similarity-based adjacency matrices
   - Called by spatio-temporal models to define rider-to-rider connections

3. **Model Artifacts**
   - XGBoost: `.json` binary format (tree structure)
   - ConvLSTM: PyTorch `.pt` state dict
   - Load with model-specific methods (XGBoost.load_model(), torch.load_state_dict())

---

## ⚠️ Common Pitfalls

1. **Forgetting to lag features** → Model trains on trivially correlated data
2. **Using speed/load_strain in X** → Inflated R² from data leakage
3. **Not grouping by courier_id** → Temporal sequence breaks across riders
4. **Missing median fillna** → NaN values cause training failures
5. **Wrong feature dimensions for ConvLSTM** → Expect (batch, time, H, W, features) tensor format

---

## 📖 Reference Documents
- `README.md` - Project overview, data descriptions, quick start
- `plan/时空预测模型.md` - STGCN architecture details (Chinese)
- `plan/数据预处理.md` - Preprocessing logic documentation
- `TECHNICAL_ROUTE.md` - (if exists) Deep technical details
