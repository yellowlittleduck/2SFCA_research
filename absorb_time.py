'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-05-03 16:44:06
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-05-03 20:01:49
FilePath: \research4\absorb_time.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import os

# ========== 全局参数 ==========
PROJ_EPSG = 32650              # 杭州的平面坐标系 EPSG
SEARCH_RADIUS_M = 10000        # 吸附的最大容忍距离 (米)
WALKING_SPEED = 1.0         # 步行速度：1.0 m/s (约 3.6 km/h，适合老年人)

# ========== 路径配置 ==========
# 替换为你实际的文件路径
pop_grid_path = r'E:\research4\hangzhou_500data 2\hangzhou_popdata.shp'      # 500x500m 人口网格
subway_station_path = r'E:\research4\metro\hangzhou_metro_stops_fixed.shp' # 地铁站数据
road_graph_path = r'E:\research4\studyarea_data\Processed_Fixed\fixed_road_graph.graphml' # 之前生成的路网
res_csv_path = r'E:\research4\studyarea_data\pop_to_subway_accessibility.csv'

print("1. 正在加载数据与路网...")
pop_gdf = gpd.read_file(pop_grid_path)
subway_gdf = gpd.read_file(subway_station_path)

# 强制统一坐标系到平面投影，确保距离计算的单位是“米”
if pop_gdf.crs is None or pop_gdf.crs.to_epsg() != PROJ_EPSG:
    pop_gdf = pop_gdf.to_crs(epsg=PROJ_EPSG)
if subway_gdf.crs is None or subway_gdf.crs.to_epsg() != PROJ_EPSG:
    subway_gdf = subway_gdf.to_crs(epsg=PROJ_EPSG)

# 覆盖 osmnx 默认的布尔类型转换，防止 SHP 中的字符串值（如 'B'）被强制转 bool 报错
_bool_attrs = ['oneway', 'bridge', 'tunnel', 'area', 'junction', 'access', 'expressway', 'toll']
_edge_dtypes = {attr: str for attr in _bool_attrs}
G = ox.load_graphml(road_graph_path, edge_dtypes=_edge_dtypes)

# 将 time_m 设为图的边权重
if G.is_multigraph():
    attr_dict = { (u, v, k): float(data['time_m']) for u, v, k, data in G.edges(keys=True, data=True) if 'time_m' in data }
else:
    attr_dict = { (u, v): float(data['time_m']) for u, v, data in G.edges(data=True) if 'time_m' in data }
nx.set_edge_attributes(G, attr_dict, name='time_m')

print("2. 正在提取多边形中心点 (Centroids)...")
# 不管你的原始数据是面还是点，统一提取质心
pop_centroids = pop_gdf.geometry.centroid
subway_centroids = subway_gdf.geometry.centroid

print("3. 正在将网格和地铁站吸附到路网节点...")
# --- 吸附人口网格中心点 ---
x_pop, y_pop = pop_centroids.x.values, pop_centroids.y.values
n_pop, d_pop = ox.distance.nearest_nodes(G, x_pop, y_pop, return_dist=True)

# 过滤掉离路网实在太远（比如江中心、深山）的无效网格
n_pop = np.where(np.array(d_pop) <= SEARCH_RADIUS_M, n_pop, -1)
pop_gdf["node"] = n_pop
pop_gdf["node_dist_m"] = d_pop

# --- 吸附地铁站中心点 ---
x_sub, y_sub = subway_centroids.x.values, subway_centroids.y.values
n_sub, d_sub = ox.distance.nearest_nodes(G, x_sub, y_sub, return_dist=True)

# 构建地铁站节点→接驳步行时间的映射（同一节点可能对应多个地铁站，取最短）
n_sub_arr = np.where(np.array(d_sub) <= SEARCH_RADIUS_M, n_sub, -1)
subway_access_time = {}  # node_id -> 步行接驳时间(分钟)
for i, (node_id, dist_m) in enumerate(zip(n_sub_arr, d_sub)):
    if node_id == -1:
        continue
    t = dist_m / WALKING_SPEED / 60
    if node_id not in subway_access_time or t < subway_access_time[node_id]:
        subway_access_time[node_id] = t

# 获取所有成功吸附的地铁站路网节点列表 (去除重复节点)
subway_nodes = list(subway_access_time.keys())

print(f"   => 成功吸附了 {len(subway_nodes)} 个独立的地铁站接入节点。")

print("4. 正在执行多源 Dijkstra 寻路计算...")
# 魔法发生在这里：把所有地铁站作为起点，算法会计算全路网每个节点到【最近一个地铁站】的最短时间
lengths = nx.multi_source_dijkstra_path_length(G, sources=subway_nodes, weight='time_m')

# 把计算出的纯路网时间映射回人口网格，并叠加地铁站端接驳步行时间
# total = 路网时间 + 地铁站到路网节点的步行时间
pop_gdf["network_time_m"] = pop_gdf["node"].map(
    lambda x: lengths.get(x, float('inf')) + subway_access_time.get(x, 0) if x != -1 else float('inf')
)

print("5. 正在叠加最后一公里接驳步行时间...")
# 时间(分钟) = 距离(米) / 步行速度(米/秒) / 60
pop_gdf["access_time_m"] = (pop_gdf["node_dist_m"] / WALKING_SPEED) / 60

# 总通行时间 = 路网上走的时间 + 从网格中心走到路网的时间
pop_gdf["nearest_subway_time"] = pop_gdf["network_time_m"] + pop_gdf["access_time_m"]

# 将无法到达（inf）的值替换为空值，方便后续统计
pop_gdf["nearest_subway_time"] = pop_gdf["nearest_subway_time"].replace([np.inf, -np.inf], np.nan)

print(f"6. 计算完成！正在保存结果至: {res_csv_path}")
# 删除不需要导出的几何列（如果你想在 QGIS 里看结果，可以改成 pop_gdf.to_file 保存为 shp）
df_export = pd.DataFrame(pop_gdf.drop(columns='geometry'))
df_export.to_csv(res_csv_path, index=False, encoding='utf-8-sig')

print("✅ 所有流程处理完毕！")