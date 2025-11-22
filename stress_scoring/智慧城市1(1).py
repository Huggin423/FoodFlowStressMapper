import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ---------------------- 1. 数据读取与基础预处理（保留稳定性）----------------------
df = pd.read_csv(
    "data/processed/preprocessed_data(28_days)/rider_features_20200201.csv",
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
df_route['peak_congestion'] = df_route['congestion_index'] * (df_route['time_period'].isin(['lunch_peak', 'dinner_peak']).astype(int))  # 高峰拥堵系数

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
    # 1) 分动作基础压力
    act_type = row['action_type']
    weights = action_weights.get(act_type, action_weights['ASSIGN'])  # 兜底
    base_stress = sum(row[feat] * w for feat, w in weights.items())
    
    # 2) Route累积系数修正（层级联动）
    route_coeff = row['route_coeff'] if not pd.isna(row['route_coeff']) else 1.0
    
    # 3) 时序衰减系数（越往后的动作，疲劳累积，系数越高：1.0~1.5）
    time_coeff = 1.0 + 0.5 * row['act_order_norm']
    
    # 4) 最终压力（放大到0~12分，截断超界值）
    final_stress = base_stress * route_coeff * time_coeff * 12
    return min(final_stress, 12.0)  # 截断超过12分的部分

df_action['stress_score'] = df_action.apply(calculate_action_stress, axis=1)

# ---------------------- 5. 压力分级与结果优化 ----------------------
# 统一分级标准（0~12分映射到低/中/高压力）
def classify_stress(score):
    if score < 4:
        return '低压力'
    elif score < 8:
        return '中压力'
    else:
        return '高压力'

df_route['stress_level'] = df_route['stress_score'].apply(classify_stress)
df_action['stress_level'] = df_action['stress_score'].apply(classify_stress)

# 层级联动修正：Action_Point 峰值反哺 Route 压力（让Route压力更贴合局部峰值）
action_peak = df_action.groupby('route_id')['stress_score'].max().reset_index()
action_peak.columns = ['route_id', 'action_stress_peak']
df_route = pd.merge(df_route, action_peak, on='route_id', how='left')
df_route['action_stress_peak'] = df_route['action_stress_peak'].fillna(0)

# 修正 Route 压力（融合全局累积+局部峰值）
df_route['stress_score_final'] = (
    0.7 * df_route['stress_score'] + 0.3 * df_route['action_stress_peak']
).clip(0, 12)  # 限制范围0~12
df_route['stress_level_final'] = df_route['stress_score_final'].apply(classify_stress)

# ---------------------- 6. 清理重复列+合并数据（解决报错核心）----------------------
# 6.1 清理 Route 数据：只保留最终需要的列，删除中间列避免冲突
df_route_final = df_route[['route_id', 'courier_id', 'date', 'feature_type', 'continuous_orders',
                           'time_period', 'load_intensity', 'congestion_index', 'stress_score_final',
                           'stress_level_final']].rename(columns={
                               'stress_score_final': 'stress_score',
                               'stress_level_final': 'stress_level'
                           })

# 6.2 清理 Action_Point 数据：只保留最终需要的列
df_action_final = df_action[['route_id', 'courier_id', 'date', 'feature_type', 'act_pt_id',
                             'act_time', 'act_order', 'action_type', 'stress_score', 'stress_level']]

# 6.3 确保列名完全一致（合并的前提）
print("Route 列名：", df_route_final.columns.tolist())
print("Action 列名：", df_action_final.columns.tolist())

# 6.4 合并数据（ignore_index=True 确保索引唯一）
df_result = pd.concat([df_route_final, df_action_final], ignore_index=True)

# ---------------------- 7. 结果输出与验证 ----------------------
# 打印关键结果
print("\n=== Route 压力结果（前10条，按最终分数排序）===")
print(df_route_final[['route_id', 'courier_id', 'continuous_orders', 'time_period', 'stress_score', 'stress_level']].sort_values('stress_score', ascending=False).head(10).round(2))

print("\n=== Action_Point 压力结果（按动作类型分组）===")
print(df_action_final.groupby('action_type')['stress_score'].agg(['mean', 'max', 'min']).round(2))

print("\n=== 层级联动效果验证（Route系数 vs Action压力）===")
link_verify = df_action.groupby('route_coeff')['stress_score'].mean().reset_index().round(2)
link_verify = link_verify.sort_values('route_coeff')  # 按系数排序，更易观察趋势
print(link_verify)

# 保存完整结果
df_result.to_csv("rider_stress_hierarchical_final.csv", index=False, encoding='utf-8-sig')
print("\n最终结果已保存到 rider_stress_hierarchical_final.csv")