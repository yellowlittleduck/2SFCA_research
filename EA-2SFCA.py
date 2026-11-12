import numpy as np
import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm

# ========== 路径配置 ==========
POP_GRID_PATH = r'E:\research4\hangzhou_500data 2\hangzhou_popdata.shp'
SUBWAY_PATH = r'E:\research4\metro\hangzhou_metro_stops_fixed.shp'
OD_DIR = r'E:\research4\studyarea_data\OD_matrix'
OUTPUT_DIR = r'E:\research4\studyarea_data\EA_2SFCA'

POP_ID_COL = 'OBJECTID'
SUBWAY_ID_COL = 'stop_id'
POP_COL = 'people'        # 人口SHP中的人口列名
T_THRESHOLD = 10

# ========== 核心衰减函数 ==========
def gaussian_decay(t, t0):
    """
    高斯时间衰减函数
    t: 实际通行时间 (分钟)
    t0: 极限搜寻时间阈值 (分钟)
    """
    return (np.exp(-0.5 * (t / t0)**2) - np.exp(-0.5)) / (1 - np.exp(-0.5))


# ========== EA-2SFCA 算法核心 ==========
def calculate_EA_2SFCA(od_df, pop_dict, supply_dict, T_threshold):
    """
    执行 EA-2SFCA 计算
    
    参数:
    - od_df: OD矩阵, 列 ['pop_id', 'subway_id', 'travel_time']
    - pop_dict: {pop_id: 人口数}
    - supply_dict: {subway_id: 供给容量}
    - T_threshold: 时间阈值 (分钟)
    """
    print(f"EA-2SFCA 计算 (阈值: {T_threshold} 分钟)...")
    print(f"   人口网格数: {len(pop_dict)}, 地铁站数: {len(supply_dict)}, OD对数: {len(od_df)}")
    
    valid_od = od_df[od_df['travel_time'] <= T_threshold].copy()
    print(f"   阈值内OD对: {len(valid_od)}")
    
    # Step 1: 计算每个地铁站的供需比 Rj
    # Rj = Sj / Σ(Pk × W(Tkj))  其中Pk只取t<15min内可达的人口
    DEMAND_CUTOFF = 15  # 需求端截止时间（分钟）
    print(f"Step 1: 计算供需比 (Rj), 需求端截止: {DEMAND_CUTOFF} min...")
    R_dict = {}
    for subway_id, group in tqdm(valid_od.groupby('subway_id'), desc="Step 1"):
        if subway_id not in supply_dict:
            continue
        
        supply_capacity = supply_dict[subway_id]
        weighted_demand = 0.0
        
        for _, row in group.iterrows():
            t = row['travel_time']
            if t >= DEMAND_CUTOFF:
                continue  # Pk只取t<15min的人口
            pop_id = row['pop_id']
            if pop_id in pop_dict:
                weighted_demand += pop_dict[pop_id] * gaussian_decay(t, T_threshold)
        
        R_dict[subway_id] = supply_capacity / weighted_demand if weighted_demand > 0 else 0.0
    
    # 诊断Rj
    if R_dict:
        r_values = list(R_dict.values())
        print(f"   Rj统计: n={len(r_values)}, min={min(r_values):.6e}, "
              f"mean={np.mean(r_values):.6e}, max={max(r_values):.6e}")
        # 打印前5个Rj
        for i, (sid, rj) in enumerate(R_dict.items()):
            if i >= 5: break
            print(f"   Rj样本: subway_id={sid}, Rj={rj:.6e}, capacity={supply_dict.get(sid, '?')}")

    # Step 2: 计算每个人口网格的可达性 Ai
    print("Step 2: 计算可达性 (Ai)...")
    A_dict = {p_id: 0.0 for p_id in pop_dict.keys()}
    
    for pop_id, group in tqdm(valid_od.groupby('pop_id'), desc="Step 2"):
        score = 0.0
        for _, row in group.iterrows():
            if row['subway_id'] in R_dict:
                score += R_dict[row['subway_id']] * gaussian_decay(row['travel_time'], T_threshold)
        A_dict[pop_id] = score

    # 诊断
    scores = list(A_dict.values())
    non_zero = sum(1 for v in scores if v > 0)
    print(f"   可达性统计: 非零={non_zero}/{len(scores)}, "
          f"min={min(scores):.6f}, max={max(scores):.6f}, mean={np.mean(scores):.6f}")

    return pd.DataFrame(list(A_dict.items()), columns=['pop_id', 'accessibility_score'])

def main(time_str, T_threshold=T_THRESHOLD):
    """对指定时段运行EA-2SFCA"""
    od_path = os.path.join(OD_DIR, f'od_matrix_f{time_str}.csv')
    output_path = os.path.join(OUTPUT_DIR, f'EA_2SFCA_f{time_str}_f{T_threshold}.csv')
    
    if not os.path.exists(od_path):
        print(f"OD矩阵不存在: {od_path}，跳过")
        return
    
    print(f"\n{'='*60}")
    print(f"时段: {time_str}")
    print(f"{'='*60}")
    
    # 1. 读取OD矩阵
    od_df = pd.read_csv(od_path)
    if od_df.empty:
        print("OD矩阵为空，跳过")
        return
    
    od_df['pop_id'] = od_df['pop_id'].astype(str)
    od_df['subway_id'] = od_df['subway_id'].astype(str)
    print(f"OD矩阵: {len(od_df)} 条, 唯一pop={od_df['pop_id'].nunique()}, 唯一subway={od_df['subway_id'].nunique()}")
    
    # 2. 读取人口数据
    pop_gdf = gpd.read_file(POP_GRID_PATH)
    print(f"人口SHP列名: {pop_gdf.columns.tolist()}")
    print(f"人口SHP前3行:\n{pop_gdf.drop(columns='geometry', errors='ignore').head(3)}")
    
    pop_gdf['pop_id'] = pop_gdf[POP_ID_COL].astype(str)
    
    # 检测人口列名
    if POP_COL not in pop_gdf.columns:
        for alt in ['people', 'population', 'pop', 'POP', '人口', 'residents', 'popu', 'POP20']:
            if alt in pop_gdf.columns:
                print(f"人口列名: '{alt}' (非默认'{POP_COL}')")
                pop_gdf[POP_COL] = pop_gdf[alt]
                break
        else:
            num_cols = pop_gdf.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c not in ['OBJECTID', 'pop_id', 'Shape_Area', 'Shape_Leng']]
            if num_cols:
                print(f"警告：未找到'{POP_COL}'列，使用'{num_cols[0]}'作为人口列")
                pop_gdf[POP_COL] = pop_gdf[num_cols[0]]
            else:
                print(f"错误：无法确定人口列！可用列: {pop_gdf.columns.tolist()}")
                return
    
    pop_dict = pop_gdf.set_index('pop_id')[POP_COL].to_dict()
    pop_values = list(pop_dict.values())
    print(f"人口网格: {len(pop_dict)} 个, 总人口: {sum(pop_values):.0f}")
    print(f"人口值样本(min/mean/max): {min(pop_values):.1f} / {np.mean(pop_values):.1f} / {max(pop_values):.1f}")
    
    # 3. 读取地铁站供给数据
    supply_gdf = gpd.read_file(SUBWAY_PATH)
    supply_gdf['subway_id'] = supply_gdf[SUBWAY_ID_COL].astype(str)
    if 'capacity' not in supply_gdf.columns:
        supply_gdf['capacity'] = 1.0
        print("地铁站capacity: 1.0 (等权Sj=1)")
    supply_dict = supply_gdf.set_index('subway_id')['capacity'].to_dict()
    print(f"地铁站: {len(supply_dict)} 个, capacity范围: "
          f"{min(supply_dict.values()):.1f} ~ {max(supply_dict.values()):.1f}")
    
    # 4. 执行EA-2SFCA
    result = calculate_EA_2SFCA(od_df, pop_dict, supply_dict, T_threshold=T_THRESHOLD)
    
    # 5. 保存（确保CSV为数据格式，避免科学计数法）
    result['pop_id'] = pd.to_numeric(result['pop_id'], errors='coerce').astype('Int64')
    # 格式化为固定小数，避免2e-05这样的科学计数法
    result['accessibility_score'] = result['accessibility_score'].map(lambda x: f'{x:.8f}')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"已保存: {output_path} ({len(result)} 条记录)")


if __name__ == "__main__":
    TIME_SLOTS = [f"{h:02d}0000" for h in range(6, 19)]
    
    print(f"将对 {len(TIME_SLOTS)} 个时段运行EA-2SFCA")
    for ts in TIME_SLOTS:
        main(ts)
    
    print(f"\n全部完成！共处理 {len(TIME_SLOTS)} 个时段")
