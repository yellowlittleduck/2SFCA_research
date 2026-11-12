import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from Impedance_coefficient import SPEED_PARAMS

# ========== 全局参数 ==========
PROJ_EPSG = 32650
SEARCH_RADIUS_M = 10000
T_THRESHOLD_MINUTES = 90

# 时间和人群配置（需与 roadmap_graph.py 一致）
DEFAULT_TIME_STR = "060000"
DEFAULT_GROUP_TYPE = "non_vulnerable"

# ========== 路径配置 ==========
SHADOW_INPUT_PATH = r"E:\research4\hangzhou_shade_roadmap"
POP_ID_COL = 'OBJECTID'
SUBWAY_ID_COL = 'stop_id'

pop_grid_path = r'E:\research4\hangzhou_500data 2\hangzhou_popdata.shp'
subway_station_path = r'E:\research4\metro\hangzhou_metro_stops_fixed.shp'
road_graph_path = fr'E:\research4\studyarea_data\fixed_roadmap\fixed_road_graph_f{DEFAULT_TIME_STR}.graphml'
res_csv_path = fr'E:\research4\studyarea_data\od_matrix_f{DEFAULT_TIME_STR}.csv'


def calculate_walking_time(distance_m, group_type="non_vulnerable"):
    """计算简单步行接驳时间（不受阴影影响）"""
    speed = SPEED_PARAMS.get(group_type, 1.0)
    return (distance_m / speed) / 60


def main(time_str=DEFAULT_TIME_STR, group_type=DEFAULT_GROUP_TYPE):
    """生成OD矩阵主函数"""
    
    # 根据时间参数确定路网图路径
    graph_path = fr'E:\research4\studyarea_data\fixed_roadmap\fixed_road_graph_f{time_str}.graphml'
    output_path = fr'E:\research4\studyarea_data\OD_matrix\od_matrix_f{time_str}.csv'
    
    print(f"1. 正在加载数据与路网 (时间: {time_str}, 人群: {group_type})...")
    pop_gdf = gpd.read_file(pop_grid_path)
    subway_gdf = gpd.read_file(subway_station_path)

    if pop_gdf.crs is None or pop_gdf.crs.to_epsg() != PROJ_EPSG:
        pop_gdf = pop_gdf.to_crs(epsg=PROJ_EPSG)
    if subway_gdf.crs is None or subway_gdf.crs.to_epsg() != PROJ_EPSG:
        subway_gdf = subway_gdf.to_crs(epsg=PROJ_EPSG)

    # 加载路网图
    _bool_attrs = ['oneway', 'bridge', 'tunnel', 'area', 'junction', 'access', 'expressway', 'toll']
    _edge_dtypes = {attr: str for attr in _bool_attrs}
    G = ox.load_graphml(graph_path, edge_dtypes=_edge_dtypes)

    # 确保 time_m 属性为 float
    if G.is_multigraph():
        attr_dict = {(u, v, k): float(data['time_m']) 
                     for u, v, k, data in G.edges(keys=True, data=True) if 'time_m' in data}
    else:
        attr_dict = {(u, v): float(data['time_m']) 
                     for u, v, data in G.edges(data=True) if 'time_m' in data}
    nx.set_edge_attributes(G, attr_dict, name='time_m')

    print(f"   路网: {len(G.nodes)} 节点, {len(G.edges)} 边")

    print("2. 正在执行节点吸附...")
    pop_centroids = pop_gdf.geometry.centroid
    subway_centroids = subway_gdf.geometry.centroid

    # 人口节点吸附
    x_pop, y_pop = pop_centroids.x.values, pop_centroids.y.values
    n_pop, d_pop = ox.distance.nearest_nodes(G, x_pop, y_pop, return_dist=True)

    pop_node_dict = {}
    for i, (idx, row) in enumerate(pop_gdf.iterrows()):
        node_id = n_pop[i]
        dist_m = d_pop[i]
        if dist_m <= SEARCH_RADIUS_M:
            if node_id not in pop_node_dict:
                pop_node_dict[node_id] = []
            pop_node_dict[node_id].append({
                'pop_id': row[POP_ID_COL],
                'access_time_m': calculate_walking_time(dist_m, group_type)
            })

    # 地铁站吸附
    x_sub, y_sub = subway_centroids.x.values, subway_centroids.y.values
    n_sub, d_sub = ox.distance.nearest_nodes(G, x_sub, y_sub, return_dist=True)

    subway_list = []
    for i, (idx, row) in enumerate(subway_gdf.iterrows()):
        node_id = n_sub[i]
        dist_m = d_sub[i]
        if dist_m <= SEARCH_RADIUS_M:
            subway_list.append({
                'subway_id': row[SUBWAY_ID_COL],
                'node': node_id,
                'access_time_m': calculate_walking_time(dist_m, group_type)
            })

    print(f"   匹配 {len(pop_node_dict)} 个人口节点, {len(subway_list)} 个地铁站")

    print(f"3. 正在生成OD矩阵 (阈值: {T_THRESHOLD_MINUTES} 分钟)...")
    od_records = []

    subway_by_node = defaultdict(list)
    for sub in subway_list:
        subway_by_node[sub['node']].append(sub)

    for sub_node, subs in tqdm(subway_by_node.items(), desc="计算辐射范围"):
        min_acc_time = min(s['access_time_m'] for s in subs)
        search_limit = T_THRESHOLD_MINUTES - min_acc_time
        if search_limit <= 0:
            continue

        lengths = nx.single_source_dijkstra_path_length(
            G, source=sub_node, cutoff=search_limit, weight='time_m'
        )

        for reached_node, network_time in lengths.items():
            if reached_node in pop_node_dict:
                for p_info in pop_node_dict[reached_node]:
                    for sub in subs:
                        total_time = sub['access_time_m'] + network_time + p_info['access_time_m']
                        if total_time <= T_THRESHOLD_MINUTES:
                            od_records.append({
                                'pop_id': p_info['pop_id'],
                                'subway_id': sub['subway_id'],
                                'travel_time': round(total_time, 2)
                            })

    print(f"4. 生成 {len(od_records)} 对OD连接，去重中...")
    od_df = pd.DataFrame(od_records)
    od_df = od_df.sort_values('travel_time').drop_duplicates(
        subset=['pop_id', 'subway_id'], keep='first'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    od_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"OD矩阵已保存: {output_path} ({len(od_df)} 条记录)")


if __name__ == "__main__":
    TIME_SLOTS = [f"{h:02d}0000" for h in range(6, 19)]
    
    print(f"将对 {len(TIME_SLOTS)} 个时段生成OD矩阵")
    for ts in TIME_SLOTS:
        main(time_str=ts)
    
    print(f"\n全部完成！共处理 {len(TIME_SLOTS)} 个时段")
