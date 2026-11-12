import numpy as np
import pandas as pd
import os

# CUDA加速检查
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

# ========== 人群参数 ==========
SPEED_PARAMS = {
    "vulnerable": 1.18,
    "non_vulnerable": 1.43,
    "child": 0.8,
    "elderly": 1.18,
}

ALPHA_PARAMS = {
    "vulnerable": 1.0,
    "non_vulnerable": 1.0,
    "child": 1.5,
    "elderly": 1.5,
}

# ========== 阴影数据缓存 ==========
_shadow_data_cache = {}


def load_shadow_data(shadow_path, time_str):
    """
    加载指定时间的阴影数据并缓存
    
    参数：
    - shadow_path: 阴影数据目录路径
    - time_str: 时间字符串，如 "060000"
    
    返回：
    - DataFrame: 包含 OSM_ID 和 MEAN 列
    """
    global _shadow_data_cache
    
    cache_key = f"{shadow_path}_{time_str}"
    if cache_key in _shadow_data_cache:
        return _shadow_data_cache[cache_key]
    
    file_path = os.path.join(shadow_path, f"Zonal_{time_str}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"阴影数据文件不存在: {file_path}")
    
    # 尝试不同编码
    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取阴影数据 {time_str}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError(f"无法读取阴影数据文件: {file_path}")
    
    if 'OSM_ID' not in df.columns or 'MEAN' not in df.columns:
        raise ValueError(f"阴影数据缺少 OSM_ID 或 MEAN 列")
    
    # 预创建字符串列用于匹配
    df['OSM_ID_str'] = df['OSM_ID'].astype(str)
    
    _shadow_data_cache[cache_key] = df
    return df


def get_shadow_ratio(shadow_path, osm_id, time_str):
    """
    获取指定路段在指定时间的阴影比例
    
    参数：
    - shadow_path: 阴影数据目录路径
    - osm_id: 路段ID
    - time_str: 时间字符串
    
    返回：
    - shadow_ratio: [0, 1]
    """
    try:
        shadow_df = load_shadow_data(shadow_path, time_str)
        osm_id_str = str(osm_id)
        row = shadow_df[shadow_df['OSM_ID_str'] == osm_id_str]
        
        if len(row) > 0:
            return float(np.clip(row.iloc[0]['MEAN'], 0.0, 1.0))
        else:
            median_shadow = shadow_df['MEAN'].median()
            default = median_shadow if not pd.isna(median_shadow) else 0.5
            print(f"警告：未找到OSM_ID {osm_id} 的阴影数据，使用默认值 {default:.3f}")
            return default
    except Exception as e:
        print(f"警告：获取阴影数据失败 (OSM_ID: {osm_id}): {e}")
        return 0.5


def calculate_impedance_single(edge_length, shadow_ratio, group_type, gamma=0.16):
    """
    计算单条边的环境阻抗通行时间
    
    公式：c_e^*(t) = (length / speed) * (1 + γ(1 - S_e(t))^α) / 60
    
    参数：
    - edge_length: 边长度（米）
    - shadow_ratio: 阴影比例 [0, 1]
    - group_type: 人群类型
    - gamma: 环境影响强度参数
    
    返回：
    - adjusted_time_min: 通行时间（分钟）
    """
    speed = SPEED_PARAMS.get(group_type, 1.0)
    alpha = ALPHA_PARAMS.get(group_type, 1.0)
    shadow_ratio = np.clip(shadow_ratio, 0.0, 1.0)
    
    c_e = edge_length / speed  # 基础通行时间（秒）
    adjustment = 1 + gamma * (1 - shadow_ratio) ** alpha
    return (c_e * adjustment) / 60.0  # 转换为分钟


def calculate_impedance_batch(edges_df, shadow_path, time_str, group_type, osm_id_col="osm_id", length_col="length", gamma=0.16):
    """
    批量计算环境阻抗通行时间（自动选择CPU/CUDA）
    
    参数：
    - edges_df: GeoDataFrame，必须包含 osm_id 和 length 列
    - shadow_path: 阴影数据目录路径
    - time_str: 时间字符串
    - osm_id_col: OSM_ID列名
    - length_col: 长度列名
    - group_type: 人群类型
    - gamma: 环境影响强度参数
    
    返回：
    - edges_df: 添加了 time_min 列的 DataFrame
    """
    speed = SPEED_PARAMS.get(group_type, 1.0)
    alpha = ALPHA_PARAMS.get(group_type, 1.0)
    
    # 加载阴影数据并合并
    shadow_df = load_shadow_data(shadow_path, time_str)
    
    # 创建匹配键
    df = edges_df.copy()
    df['_osm_id_str'] = df[osm_id_col].astype(str)
    
    # 合并阴影比例
    df = df.merge(
        shadow_df[['OSM_ID_str', 'MEAN']].rename(columns={'MEAN': 'shadow_ratio'}),
        left_on='_osm_id_str', right_on='OSM_ID_str', how='left'
    )
    
    # 统计缺失的阴影数据
    missing_shadow_mask = df['shadow_ratio'].isna()
    n_missing_shadow = missing_shadow_mask.sum()
    if n_missing_shadow > 0:
        missing_shadow_ids = df.loc[missing_shadow_mask, osm_id_col].tolist()
        median_shadow = shadow_df['MEAN'].median()
        default_shadow = median_shadow if not pd.isna(median_shadow) else 0.5
        print(f"警告：{n_missing_shadow} 条边缺少阴影数据，使用默认值 {default_shadow:.3f}")
        print(f"  缺失阴影的osm_id (前20个): {missing_shadow_ids[:20]}")
    else:
        default_shadow = 0.5
    df['shadow_ratio'] = df['shadow_ratio'].fillna(default_shadow)
    df['shadow_ratio'] = df['shadow_ratio'].clip(0.0, 1.0)
    
    # 统计缺失的长度数据
    missing_length_mask = df[length_col].isna()
    n_missing_length = missing_length_mask.sum()
    if n_missing_length > 0:
        missing_length_ids = df.loc[missing_length_mask, osm_id_col].tolist()
        median_length = df[length_col].median()
        default_length = median_length if not pd.isna(median_length) else 100.0
        print(f"警告：{n_missing_length} 条边缺少长度数据，使用默认值 {default_length:.2f}米")
        print(f"  缺失长度的osm_id (前20个): {missing_length_ids[:20]}")
    lengths = df[length_col].fillna(df[length_col].median()).values
    
    # 向量化计算
    if CUDA_AVAILABLE:
        print(f"使用CUDA加速计算 {len(df)} 条边...")
        lengths_gpu = cp.asarray(lengths, dtype=cp.float64)
        shadow_gpu = cp.asarray(df['shadow_ratio'].values, dtype=cp.float64)
        
        c_e = lengths_gpu / speed
        adjustment = 1 + gamma * (1 - shadow_gpu) ** alpha
        time_min = cp.asnumpy(c_e * adjustment / 60.0)
    else:
        print(f"使用CPU计算 {len(df)} 条边...")
        c_e = lengths / speed
        adjustment = 1 + gamma * (1 - df['shadow_ratio'].values) ** alpha
        time_min = c_e * adjustment / 60.0
    
    # 写回结果
    edges_df['time_min'] = np.round(time_min, 2)

    ###
    # print("\n==============================")
    # print(f"Time: {time_str}")
    # print("==============================")



    # print("\n1. Shadow ratio")
    # print(df["shadow_ratio"].describe())

    # print("\n2. Impedance adjustment")
    # adj = pd.Series(adjustment)
    # print(adj.describe())

    # print("\n3. Travel time (min)")
    # tt = pd.Series(time_min)
    # print(tt.describe())

    # print("\n4. Extra impedance (%)")
    # extra = (adj - 1) * 100
    # print(extra.describe())

    # print("\n5. Quantiles of shadow ratio")
    # print(df["shadow_ratio"].quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))

    # print("\n6. Quantiles of adjustment")
    # print(adj.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))

    # print("\n7. Max increase")
    # print(f"Maximum impedance factor : {adj.max():.4f}")
    # print(f"Mean impedance factor    : {adj.mean():.4f}")
    # print(f"Mean extra cost          : {(adj.mean()-1)*100:.2f}%")
    # print(f"Maximum extra cost       : {(adj.max()-1)*100:.2f}%")
    
    # print(df["length"].describe())
    # print(df["length"].quantile([0.5,0.75,0.9,0.95,0.99]))
    ###

    
    # 清理临时列
    valid_times = edges_df['time_min'].dropna()
    print(f"计算完成: {len(valid_times)}/{len(edges_df)} 条边, 平均 {valid_times.mean():.2f} 分钟")
    
    return edges_df

