"""
骑手配送压力预测 - 数据预处理与DSI计算模块 (v2.0 增强版)
修复内容:
  1. 日期解析bug: 从GeoJSON文件名正确提取日期
  2. 天气字段bug: 使用模糊匹配改善天气分类
  3. DSI计算: 基于物理约束的配送紧迫指数
  4. [新增] start_hour: 提取每条路线的起始小时，支持更细粒度的STGCN聚合
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path
import warnings
import math
warnings.filterwarnings('ignore')


class DeliveryRiderDataPreprocessor:
    def __init__(self, data_dir):
        """初始化数据预处理器"""
        self.data_dir = Path(data_dir)
        self.features_df = pd.DataFrame()
        
    def load_geojson_files(self):
        """加载所有GeoJSON文件"""
        geojson_files = list(self.data_dir.glob("DeliveryRoutes_*.geojson"))
        print(f"找到 {len(geojson_files)} 个GeoJSON文件")
        return sorted(geojson_files)
    
    def extract_date_from_filename(self, filename):
        """从文件名中提取日期"""
        try:
            stem = Path(filename).stem
            date_str = stem.replace('DeliveryRoutes_', '')
            
            if len(date_str) == 8 and date_str.isdigit():
                datetime.strptime(date_str, '%Y%m%d')
                return date_str
            else:
                print(f"  [WARN] 文件名日期格式异常: {stem}")
                return None
        except Exception as e:
            print(f"  [WARN] 日期提取失败 {filename}: {e}")
            return None
    
    def parse_complex_fields(self, properties):
        """手动解析复杂字段（列表类型）"""
        def safe_json_parse(value, default=[]):
            if value is None:
                return default
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    value = value.replace("'", '"').replace("None", "null")
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, list) else default
                except:
                    return default
            return default
        
        act_lst = safe_json_parse(properties.get('act_lst'))
        r_time_lst = safe_json_parse(properties.get('r_time_lst'))
        r_dis_lst = safe_json_parse(properties.get('r_dis_lst'))
            
        return act_lst, r_time_lst, r_dis_lst
    
    def get_route_id(self, properties):
        """安全获取route_id"""
        route_id = properties.get('route_id') or properties.get('Route_id')
        if route_id is not None:
            return str(route_id)
        return None
    
    def calculate_entropy(self, action_list):
        """计算动作列表的信息熵"""
        if not action_list or len(action_list) == 0:
            return 0
        
        try:
            actions = pd.Series(action_list)
            value_counts = actions.value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
            return entropy
        except:
            return 0
    
    def calculate_time_features(self, time_list):
        """计算时间相关特征"""
        if len(time_list) < 2:
            return 0, 1, 0
        
        try:
            valid_times = [float(t) for t in time_list if t is not None and float(t) > 0]
            if len(valid_times) < 2:
                return 0, 1, 0
                
            valid_times = sorted(valid_times)
            time_diffs = np.diff(valid_times)
            avg_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 0
            
            continuous_orders = 1
            max_continuous = 1
            for diff in time_diffs:
                if diff < 600:  # 10分钟
                    continuous_orders += 1
                    max_continuous = max(max_continuous, continuous_orders)
                else:
                    continuous_orders = 1
            
            total_duration = max(valid_times) - min(valid_times)
            task_density = len(valid_times) / (total_duration / 60) if total_duration > 0 else 0
            
            return avg_interval, max_continuous, task_density
        except:
            return 0, 1, 0
    
    def haversine_distance(self, coord1, coord2):
        """计算两个坐标点之间的haversine距离（单位：米）"""
        try:
            lat1, lon1 = math.radians(coord1[1]), math.radians(coord1[0])
            lat2, lon2 = math.radians(coord2[1]), math.radians(coord2[0])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371000 
            return c * r
        except:
            return 0
    
    def calculate_route_curvature(self, total_distance, coordinates):
        """计算路径曲折度：实际行驶距离 / 起终点直线距离"""
        if total_distance <= 0:
            return 1.0
        
        if not coordinates or len(coordinates) < 2:
            return 1.0
        
        try:
            start_coord = coordinates[0]
            end_coord = coordinates[-1]
            
            straight_distance = self.haversine_distance(start_coord, end_coord)
            
            if straight_distance > 0:
                curvature = total_distance / straight_distance
                return min(max(curvature, 1.0), 10.0)
            else:
                return 1.0
        except:
            return 1.0
    
    def extract_coordinates_from_geometry(self, geometry):
        """从geometry中提取坐标"""
        if not geometry:
            return None
        
        try:
            geom_type = geometry.get('type', '')
            coordinates = geometry.get('coordinates', [])
            
            if geom_type == 'LineString' and len(coordinates) >= 2:
                return coordinates
            elif geom_type == 'Point':
                return [coordinates]
            else:
                return None
        except:
            return None
    
    def get_center_coordinates(self, coordinates):
        """计算路线的中心点坐标 (lat, lon)"""
        if not coordinates:
            return None, None
        try:
            # coordinates通常是 [[lon, lat], [lon, lat], ...]
            coords_np = np.array(coordinates)
            if coords_np.ndim == 1: # Point case
                 mean_lon, mean_lat = coords_np[0], coords_np[1]
            elif coords_np.ndim == 2: # LineString case
                 mean_lon = np.mean(coords_np[:, 0])
                 mean_lat = np.mean(coords_np[:, 1])
            else: # Multi-dim edge case
                 mean_lon = np.mean(coords_np[:, 0])
                 mean_lat = np.mean(coords_np[:, 1])
            return mean_lat, mean_lon
        except:
            return None, None

    def map_weather_score(self, weather_grade):
        """天气评分映射 - 使用模糊匹配"""
        if not weather_grade:
            return 0
        
        weather_str = str(weather_grade).strip().lower()
        
        # 恶劣天气关键词
        bad_weather_keywords = ['雨', '雪', '雾', '霾', '冰', '雹', '台风', '暴雨', '暴雪', 
                                'rain', 'snow', 'hail', 'storm', 'heavy']
        
        for keyword in bad_weather_keywords:
            if keyword in weather_str:
                return 2
        
        # 轻微恶劣天气
        moderate_keywords = ['阴', '多云', 'cloud', 'overcast']
        for keyword in moderate_keywords:
            if keyword in weather_str:
                return 1
        
        return 0
    
    def get_time_period(self, timestamp):
        """根据时间戳获取时间段分类"""
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            hour = dt.hour
            
            if 6 <= hour < 10:
                return "morning_peak"
            elif 11 <= hour < 13:
                return "lunch_peak"
            elif 17 <= hour < 19:
                return "dinner_peak"
            elif 10 <= hour < 17:
                return "day_off_peak"
            else:
                return "night"
        except:
            return "unknown"
    
    def safe_float_conversion(self, value, default=0.0):
        """安全转换为浮点数"""
        if value is None:
            return default
        try:
            return float(value)
        except:
            return default
    
    def process_route_level_data(self, properties, geometry=None):
        """处理路线级别的数据"""
        try:
            # 解析复杂字段
            act_lst, r_time_lst, r_dis_lst = self.parse_complex_fields(properties)
            
            # 提取坐标信息
            coordinates = self.extract_coordinates_from_geometry(geometry)
            # 计算中心点坐标 (为 STGCN 准备)
            center_lat, center_lon = self.get_center_coordinates(coordinates)
            
            # 提取基本特征
            route_id = self.get_route_id(properties)
            courier_id = properties.get('courier_id', '')
            date = properties.get('date', '')
            
            # 安全转换数值特征
            no_act = self.safe_float_conversion(properties.get('no_act', 0))
            r_dur_all = self.safe_float_conversion(properties.get('r_dur_all', 0))
            r_dis_all = self.safe_float_conversion(properties.get('r_dis_all', 0))
            no_nav = self.safe_float_conversion(properties.get('no_nav', 0))
            nav_dis = self.safe_float_conversion(properties.get('nav_dis', 0))
            nav_dur = self.safe_float_conversion(properties.get('nav_dur', 0))
            rider_lvl = self.safe_float_conversion(properties.get('rider_lvl', 0))
            rider_spd = self.safe_float_conversion(properties.get('rider_spd', 0))
            max_load = max(self.safe_float_conversion(properties.get('max_load', 1)), 1)
            
            # 环境特征 - 改进天气数据提取
            wthr_grd = properties.get('wthr_grd')
            if wthr_grd is None:
                wthr_grd = properties.get('weather_grade')  # 尝试其他可能的字段名
            
            # 只有路线级别的数据才计算这些特征
            if no_act > 0 and len(act_lst) > 0:
                # 1. 个体行为特征
                order_rate = no_act / (r_dur_all / 3600) if r_dur_all > 0 else 0
                
                avg_interval, continuous_orders, task_density = self.calculate_time_features(r_time_lst)
                
                current_load = min(no_act, max_load)
                load_intensity = current_load / max_load if max_load > 0 else 0
                
                act_entropy = self.calculate_entropy(act_lst)
                
                # 2. 时序动态特征
                avg_speed = r_dis_all / r_dur_all if r_dur_all > 0 else 0
                
                level_speed_expectation = {1: 3.5, 2: 4.0, 3: 4.5, 4: 5.0, 5: 5.5}
                expected_speed = level_speed_expectation.get(int(rider_lvl), 4.5)
                spd_dev = rider_spd - expected_speed
                
                nav_ratio = nav_dur / (r_dur_all + nav_dur) if (r_dur_all + nav_dur) > 0 else 0
                
                task_per_km = no_act / (r_dis_all / 1000) if r_dis_all > 0 else 0
                
                # 3. 空间特征 - 简化的路径曲折度计算
                route_curvature = self.calculate_route_curvature(r_dis_all, coordinates)
                
                # 4. 环境压力特征
                weather_score = self.map_weather_score(wthr_grd)
                
                # 时间段与小时提取（关键修改：支持小时级聚合）
                start_hour = -1
                if r_time_lst and len(r_time_lst) > 0:
                    try:
                        dt = datetime.fromtimestamp(float(r_time_lst[0]))
                        start_hour = dt.hour
                        time_period = self.get_time_period(r_time_lst[0])
                    except:
                        time_period = "unknown"
                else:
                    time_period = "unknown"
                
                # 区域拥堵度
                city_avg_speed = 4.0
                congestion_index = max(0, min(1, 1 - (avg_speed / city_avg_speed))) if city_avg_speed > 0 else 0
                
                # ==================== DSI计算:基于物理约束的配送紧迫指数 ====================
                # DSI = α * (Required_Speed / Traffic_Speed) + β * (Current_Load / Max_Load)
                
                # 计算Required_Speed:完成剩余订单所需的最小速度
                if r_dur_all > 0 and avg_speed > 0:
                    required_speed = r_dis_all / (r_dur_all * 0.8)  # 假设需要在80%时间内完成
                else:
                    required_speed = avg_speed
                
                # Traffic_Speed:该路段的通行速度
                traffic_speed = max(avg_speed, city_avg_speed)
                
                # 速度压力分量
                speed_strain = required_speed / traffic_speed if traffic_speed > 0 else 0
                speed_strain = min(speed_strain, 3.0)
                
                # 负载压力分量
                load_strain = current_load / max_load if max_load > 0 else 0
                
                # DSI综合指数(权重可调)
                alpha = 0.6
                beta = 0.4
                dsi = alpha * speed_strain + beta * load_strain
                dsi = min(max(dsi, 0), 2.0)  # 限制在[0, 2]
                dsi_score = dsi * 5.0  # 映射到[0, 10]分
                
            else:
                # 对于无效的路线数据,返回空特征
                order_rate = avg_interval = continuous_orders = load_intensity = act_entropy = 0
                avg_speed = spd_dev = task_density = nav_ratio = task_per_km = 0
                route_curvature = 1.0
                weather_score = congestion_index = 0
                time_period = "unknown"
                start_hour = -1
                dsi_score = speed_strain = load_strain = 0.0
            
            # 构建特征字典
            feature_dict = {
                # 基础信息
                'route_id': route_id,
                'courier_id': courier_id,
                'date': date,
                'feature_type': 'route',
                
                # 空间坐标 (新增：用于直接生成Grid ID)
                'center_lat': center_lat,
                'center_lon': center_lon,
                
                # ========== 核心目标变量:DSI (Delivery Strain Index) ==========
                'dsi': dsi_score,
                'speed_strain': speed_strain,
                'load_strain': load_strain,
                
                # 个体行为特征
                'order_rate': order_rate,
                'avg_interval': avg_interval,
                'continuous_orders': continuous_orders,
                'load_intensity': load_intensity,
                'act_entropy': act_entropy,
                
                # 时序动态特征
                'avg_speed': avg_speed,
                'spd_dev': spd_dev,
                'task_density': task_density,
                'nav_ratio': nav_ratio,
                'task_per_km': task_per_km,
                
                # 空间特征
                'route_curvature': route_curvature,
                
                # 环境压力特征
                'weather_score': weather_score,
                'time_period': time_period,
                'start_hour': start_hour, # 新增字段：小时
                'congestion_index': congestion_index,
                
                # 原始特征
                'no_act': no_act,
                'r_dur_all': r_dur_all,
                'r_dis_all': r_dis_all,
                'rider_lvl': rider_lvl,
                'rider_spd': rider_spd,
                'max_load': max_load,
                'weather_raw': str(wthr_grd)[:50] if wthr_grd else 'unknown'
            }
            
            return feature_dict
            
        except Exception as e:
            print(f"处理路线数据 {self.get_route_id(properties)} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_action_point_data(self, properties, route_cache, geometry=None):
        """处理动作点级别的数据，并关联对应的路线数据"""
        try:
            route_id = self.get_route_id(properties)
            act_pt_id = properties.get('act_pt_id', '')
            act_time = self.safe_float_conversion(properties.get('act_time', 0))
            
            # 获取对应的路线级别数据
            route_data = route_cache.get(route_id, {})
            
            if route_data and self.safe_float_conversion(route_data.get('no_act', 0)) > 0:
                # 使用路线数据计算特征
                feature_dict = self.process_route_level_data(route_data, geometry)
                if feature_dict:
                    feature_dict.update({
                        'act_pt_id': act_pt_id,
                        'act_time': act_time,
                        'action_type': properties.get('action_type', ''),
                        'feature_type': 'action_point_with_route'
                    })
                return feature_dict
            else:
                # 简化处理：如果没有路线数据，返回基本空壳（这里略去部分非核心逻辑以保持代码简洁）
                return None
                
        except Exception as e:
            return None
    
    def process_geojson_file(self, file_path):
        """处理单个GeoJSON文件"""
        print(f"正在处理文件: {file_path.name}")
        
        try:
            date_from_filename = self.extract_date_from_filename(file_path.name)
            if not date_from_filename:
                return pd.DataFrame()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features_list = []
            route_count = 0
            
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                
                if 'date' not in properties or not properties.get('date'):
                    properties['date'] = date_from_filename
                
                feature_type = properties.get('feature_type', '')
                
                if feature_type == 'route':
                    route_features = self.process_route_level_data(properties, geometry)
                    if route_features:
                        features_list.append(route_features)
                        route_count += 1
            
            result_df = pd.DataFrame(features_list)
            print(f"  处理完成: {route_count} 条路线数据")
            return result_df
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return pd.DataFrame()
    
    def process_all_data(self, output_dir="output_features"):
        """处理所有数据"""
        geojson_files = self.load_geojson_files()
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_features = []
        for file_path in geojson_files:
            print(f"\n{'='*60}")
            df = self.process_geojson_file(file_path)
            if not df.empty:
                date_str = self.extract_date_from_filename(file_path.name)
                if date_str:
                    output_file = output_path / f"rider_features_{date_str}.csv"
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"  已保存: {output_file}")
                    all_features.append(df)
        
        if all_features:
            self.features_df = pd.concat(all_features, ignore_index=True)
            print(f"\n{'='*60}")
            print(f"所有数据处理完成! 总记录数: {len(self.features_df)}")
            return self.features_df

if __name__ == "__main__":
    # 设置数据目录路径
    data_directory = "./data/raw/ODIDMob_Routes"
    preprocessor = DeliveryRiderDataPreprocessor(data_directory)
    preprocessor.process_all_data("output_features")