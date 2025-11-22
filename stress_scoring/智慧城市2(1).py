import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ---------------------- 1. 数据读取与基础预处理（改成：读取 28 天全部）----------------------
data_dir = "data/processed/preprocessed_data(28_days)"
pattern = os.path.join(data_dir, "rider_features_*.csv")
files = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(f"在 {data_dir} 下没有找到 rider_features_*.csv，请先确认预处理是否已生成 28 天特征文件。")

df_list = []
for f in files:
    print("loading:", os.path.basename(f))
    _df = pd.read_csv(
        f,
        encoding="utf-8-sig",
        dtype={
            'continuous_orders': float,
            'no_act': float,
            'act_order': float,
            'load_intensity': float,
            'congestion_index': float,
            'order_rate': float,
            'task_per_km': float
        }
    )
    df_list.append(_df)

df = pd.concat(df_list, ignore_index=True)
print("总样本量：", len(df))

# 快速填充缺失值（全局均值，避免分组覆盖）
num_cols = ['continuous_orders', 'no_act', 'act_order', 'load_intensity', 'congestion_index',
            'order_rate', 'task_per_km', 'avg_interval', 'task_density', 'nav_ratio', 'weather_score']
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# 动作类型+时段预处理（统一格式）
df['action_type'] = df['action_type'].fillna('ASSIGN').str.upper()
df['time_period'] = df['time_period'].fillna('day_off_peak')
le_time = LabelEncoder()
df['time_period_encoded'] = le_time.fit_transform(df['time_period'])

# 分离 Route 和 Action_Point 数据（提前拆分，重置索引确保唯一）
df_route = df[df['feature_type'] == 'route'].copy().reset_index(drop=True)  # 重置索引
df_action = df[df['feature_type'] == 'action_point_with_route'].copy().reset_index(drop=True)  # 重置索引

# ---------------------- 2. 特征工程（新增层级联动所需特征）----------------------
# 2.1 全局归一化（保留差异，0~1区间）
normalized_cols = [
    'load_intensity', 'continuous_orders', 'task_per_km', 'congestion_index',
    'weather_score', 'task_density', 'order_rate', 'time_period_encoded'
]
scaler = MinMaxScaler(feature_range=(0, 1))
df_route[normalized_cols] = scaler.fit_transform(df_route[normalized_cols])
df_action[normalized_cols] = scaler.transform(df_action[normalized_cols])  # 用Route的scaler保持一致

# 2.2 反向特征（时效压力）
scaler_interval = MinMaxScaler(feature_range=(0, 1))
df_route['avg_interval_rev'] = scaler_interval.fit_transform(1 - df_route[['avg_interval']])
df_action['avg_interval_rev'] = scaler_interval.transform(1 - df_action[['avg_interval']])

# 2.3 Route 累积特征（层级联动核心）
df_route['accumulation_load'] = df_route['continuous_orders'] * df_route['load_intensity']  # 连续订单×负载（累积负荷）
df_route['peak_congestion'] = df_route['congestion_index'] * (
    df_route['time_period'].isin(['lunch_peak', 'dinner_peak']).astype(int)
)  # 高峰拥堵系数

# 2.4 Action_Point 时序特征（动作顺序衰减）
df_action['act_order_norm'] = df_action['act_order'] / df_action['no_act']  # 动作顺序归一化（0~1，越往后越接近1）

# ---------------------- 3. 权重方案（分类型+分动作+层级联动）----------------------
# 3.1 Route 权重（全局行程：累积负荷+环境压力）
route_weights = {
    'accumulation_load': 0.35,    # 累积负载（核心联动特征）
    'peak_congestion': 0.25,      # 高峰拥堵（场景压力）
    'task_per_km': 0.20,          # 单位距离任务密度
    'weather_score': 0.10,        # 天气影响
    'order_rate': 0.10            # 订单频率
}

# 3.2 Action_Point 分动作权重（差异化压力源）
action_weights = {
    'ASSIGN': {  # 接单：关注订单频率+时段
        'order_rate': 0.35,
        'time_period_encoded': 0.30,
        'congestion_index': 0.15,
        'weather_score': 0.10,
        'load_intensity': 0.10
    },
    'PICKUP': {  # 取餐：关注负载+任务密度
        'load_intensity': 0.40,
        'task_density': 0.25,
        'avg_interval_rev': 0.15,
        'congestion_index': 0.10,
        'weather_score': 0.10
    },
    'DELIVERY': {  # 送达：关注时效+拥堵
        'avg_interval_rev': 0.40,
        'congestion_index': 0.25,
        'task_density': 0.15,
        'load_intensity': 0.10,
        'weather_score': 0.10
    }
}

# 3.3 层级联动系数（Route→Action_Point 修正）
def calculate_route_coeff(row):
    """Route累积系数：连续订单越多+高峰时段，系数越高（1.0~2.0）"""
    continuous_coeff = 1.0 + 0.2 * min(row['continuous_orders'], 5)  # 连续订单最多放大2倍
    peak_coeff = 1.2 if row['time_period'] in ['lunch_peak', 'dinner_peak'] else 1.0  # 高峰额外放大
    return min(continuous_coeff * peak_coeff, 2.0)  # 上限2.0，避免过度放大

df_route['route_coeff'] = df_route.apply(calculate_route_coeff, axis=1)

# 将 Route 系数映射到 Action_Point（通过 route_id 关联）
route_coeff_dict = df_route.set_index('route_id')['route_coeff'].to_dict()
df_action['route_coeff'] = df_action['route_id'].map(route_coeff_dict)

# ---------------------- 4. 压力计算（层级联动核心逻辑）----------------------
# 4.1 计算 Route 压力（全局累积压力）
df_route['stress_score'] = (
    df_route['accumulation_load'] * route_weights['accumulation_load'] +
    df_route['peak_congestion'] * route_weights['peak_congestion'] +
    df_route['task_per_km'] * route_weights['task_per_km'] +
    df_route['weather_score'] * route_weights['weather_score'] +
    df_route['order_rate'] * route_weights['order_rate']
) * 12  # 放大到0~12分，预留峰值空间

# 4.2 计算 Action_Point 压力（分动作+层级修正+时序衰减）
def calculate_action_stress(row):
    """动作点压力 = 分动作基础压力 × Route累积系数 × 时序衰减系数"""
    act_type = row['action_type']
    weights = action_weights.get(act_type, action_weights['ASSIGN'])  # 兜底
    base_stress = sum(row[feat] * w for feat, w in weights.items())
    route_coeff = row['route_coeff'] if not pd.isna(row['route_coeff']) else 1.0
    time_coeff = 1.0 + 0.5 * row['act_order_norm']  # 越靠后越累
    final_stress = base_stress * route_coeff * time_coeff * 12
    return min(final_stress, 12.0)

df_action['stress_score'] = df_action.apply(calculate_action_stress, axis=1)

# ---------------------- 5. 压力分级与结果优化 ----------------------
def classify_stress(score):
    if score < 4:
        return '低压力'
    elif score < 8:
        return '中压力'
    else:
        return '高压力'

df_route['stress_level'] = df_route['stress_score'].apply(classify_stress)
df_action['stress_level'] = df_action['stress_score'].apply(classify_stress)

# Action_Point 峰值反哺 Route
action_peak = df_action.groupby('route_id')['stress_score'].max().reset_index()
action_peak.columns = ['route_id', 'action_stress_peak']
df_route = pd.merge(df_route, action_peak, on='route_id', how='left')
df_route['action_stress_peak'] = df_route['action_stress_peak'].fillna(0)

df_route['stress_score_final'] = (
    0.7 * df_route['stress_score'] + 0.3 * df_route['action_stress_peak']
).clip(0, 12)
df_route['stress_level_final'] = df_route['stress_score_final'].apply(classify_stress)

# ---------------------- 6. 清理重复列+合并数据 ----------------------
df_route_final = df_route[['route_id', 'courier_id', 'date', 'feature_type', 'continuous_orders',
                           'time_period', 'load_intensity', 'congestion_index', 'stress_score_final',
                           'stress_level_final']].rename(columns={
                               'stress_score_final': 'stress_score',
                               'stress_level_final': 'stress_level'
                           })

df_action_final = df_action[['route_id', 'courier_id', 'date', 'feature_type', 'act_pt_id',
                             'act_time', 'act_order', 'action_type', 'stress_score', 'stress_level']]

print("Route 列名：", df_route_final.columns.tolist())
print("Action 列名：", df_action_final.columns.tolist())

df_result = pd.concat([df_route_final, df_action_final], ignore_index=True)

print("\n=== Route 压力结果（前10条，按最终分数排序）===")
print(df_route_final[['route_id', 'courier_id', 'continuous_orders', 'time_period',
                      'stress_score', 'stress_level']].sort_values('stress_score',
                      ascending=False).head(10).round(2))

print("\n=== Action_Point 压力结果（按动作类型分组）===")
print(df_action_final.groupby('action_type')['stress_score'].agg(['mean', 'max', 'min']).round(2))

print("\n=== 层级联动效果验证（Route系数 vs Action压力）===")
link_verify = df_action.groupby('route_coeff')['stress_score'].mean().reset_index().round(2)
link_verify = link_verify.sort_values('route_coeff')
print(link_verify)

df_result.to_csv("rider_stress_hierarchical_final.csv", index=False, encoding='utf-8-sig')
print("\n最终结果已保存到 rider_stress_hierarchical_final.csv")

# =================== 后半段 XGBoost 优化部分，保持不变 ===================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_csv("rider_stress_hierarchical_final.csv", encoding="utf-8-sig")

df_route = df[df['feature_type'] == 'route'].copy().reset_index(drop=True)
print(f"Route 数据量：{len(df_route)}")
df_action = df[df['feature_type'] == 'action_point_with_route'].copy()

target_y = 'stress_score'
df_route = df_route[df_route[target_y] <= df_route[target_y].quantile(0.95)]
print(f"\n使用的压力代理变量（y）：{target_y}")
print(f"y的统计信息：\n{df_route[target_y].describe().round(3)}")

candidate_features = [
    'load_intensity', 'continuous_orders', 'congestion_index',
    'task_per_km', 'weather_score', 'task_density', 'order_rate', 'time_period_encoded',
    'avg_interval_rev', 'route_coeff'
]

input_features = [feat for feat in candidate_features if feat in df_route.columns and not df_route[feat].isna().all()]
if len(input_features) < 3:
    input_features = ['load_intensity', 'continuous_orders', 'congestion_index']

df_model = df_route[input_features + [target_y]].copy().dropna()
print(f"\n最终输入特征（共{len(input_features)}个）：{input_features}")
print(f"建模数据量：{len(df_model)}")

X = df_model[input_features]
y = df_model[target_y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, shuffle=True)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=80,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    verbose=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n模型效果：")
print(f"  R²分数：{r2:.3f}")
print(f"  RMSE：{rmse:.3f}")

xgb_importance = pd.DataFrame({
    '特征名称': input_features,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False).reset_index(drop=True)
print("\n=== XGBoost自带特征重要性 ===")
print(xgb_importance.round(3))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.bar(xgb_importance['特征名称'], xgb_importance['重要性'])
plt.title("XGBoost特征重要性（影响骑手压力的核心因素）", fontsize=12)
plt.xlabel("特征名称", fontsize=10)
plt.ylabel("重要性分数", fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

total_importance = xgb_importance['重要性'].sum()
optimized_weights = dict(
    zip(xgb_importance['特征名称'], xgb_importance['重要性'] / total_importance)
)
print("\n=== 优化后的特征权重 ===")
for feat, w in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {w:.3f}")

df_route['optimized_stress_score'] = model.predict(df_route[input_features])
scaler_stress = MinMaxScaler(feature_range=(0, 12))
df_route['optimized_stress_score'] = scaler_stress.fit_transform(df_route[['optimized_stress_score']])

def classify_stress(score):
    if score < 4:
        return '低压力'
    elif score < 8:
        return '中压力'
    else:
        return '高压力'

df_route['optimized_stress_level'] = df_route['optimized_stress_score'].apply(classify_stress)

result_cols = [
    'route_id', 'courier_id', 'date', 'continuous_orders', 'time_period',
    'optimized_stress_score', 'optimized_stress_level', target_y
]
df_result = df_route[result_cols].copy()
df_result.to_csv("xgb_optimized_stress.csv", index=False, encoding='utf-8-sig')

weights_df = pd.DataFrame(list(optimized_weights.items()), columns=['特征名称', '优化后权重'])
weights_df.to_csv("xgb_optimized_weights.csv", index=False, encoding='utf-8-sig')

print("\n=== 优化后压力分数示例（前10条高压力）===")
print(df_result.sort_values('optimized_stress_score', ascending=False).head(10).round(2))

print("\n✅ 分析完成！输出文件：")
print("1. rider_stress_hierarchical_final.csv  →  28天 route+action 压力结果")
print("2. xgb_optimized_stress.csv             →  28天 route 精准压力分数（0~12分）")
print("3. xgb_optimized_weights.csv            →  XGBoost 学到的特征权重")
print("4. xgb_feature_importance.png           →  特征重要性可视化图")
