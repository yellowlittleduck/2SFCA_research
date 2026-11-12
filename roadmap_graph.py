import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from shapely import set_precision
import osmnx as ox
import numpy as np
from sklearn.neighbors import KDTree
import os
from tqdm import tqdm
from Impedance_coefficient import calculate_impedance_batch

# ========== 全局参数 ==========
PROJ_EPSG = 32650
GRID_TOL_M = 1.0

SHADOW_INPUT_PATH = r"E:\research4\hangzhou_shade_roadmap"
DEFAULT_GROUP_TYPE = "non_vulnerable"


def build_graph_from_fixed_edges(edges_gdf, time_str, 
                                  group_type=DEFAULT_GROUP_TYPE):
    """
    从已修复的 edges_fixed.shp 构建路网图
    使用环境阻抗计算通行时间
    """
    # 1. 投影转换与数据清洗
    edges = edges_gdf.copy()
    edges = edges.to_crs(epsg=PROJ_EPSG)
    edges = edges.explode(index_parts=False, ignore_index=True)
    edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty].copy()

    # 统一节点坐标精度
    edges["geometry"] = edges.geometry.apply(lambda g: set_precision(g, GRID_TOL_M))
    edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty].copy()
    edges = edges[edges.geometry.apply(lambda g: len(list(g.coords)) >= 2)].copy()

    # 2. 计算环境阻抗通行时间
    print("计算环境阻抗通行时间...")
    
    # 确保有length列（从几何计算）
    if 'length' not in edges.columns:
        edges['length'] = edges.geometry.length
    
    # 批量计算阻抗
    edges = calculate_impedance_batch(
        edges_df=edges,
        shadow_path=SHADOW_INPUT_PATH,
        time_str=time_str,
        osm_id_col="osm_id",
        length_col="length",
        group_type=group_type
    )
    
    # time_min 列即为通行时间
    edges['time_m'] = edges['time_min']

    # 3. 提取端点生成 Nodes
    def _extract_uv_points(geom):
        cs = list(geom.coords)
        if len(cs) < 2:
            return None, None
        return Point(cs[0]), Point(cs[-1])

    endpoints = edges.geometry.apply(_extract_uv_points)
    edges["u_geom"] = endpoints.apply(lambda x: x[0])
    edges["v_geom"] = endpoints.apply(lambda x: x[1])

    # 汇总所有唯一节点
    nodes_all = gpd.GeoDataFrame(
        geometry=pd.concat([edges["u_geom"], edges["v_geom"]], ignore_index=True),
        crs=edges.crs
    )
    nodes_all["geometry"] = nodes_all.geometry.apply(lambda p: set_precision(p, GRID_TOL_M))
    nodes_all["_xy"] = nodes_all.geometry.apply(lambda p: (p.x, p.y))
    nodes = nodes_all.drop_duplicates("_xy").drop(columns="_xy").reset_index(drop=True)
    
    nodes["osmid"] = nodes.index.astype(int)
    nodes = nodes.set_index("osmid")
    nodes["x"] = nodes.geometry.x
    nodes["y"] = nodes.geometry.y

    # 构建快速查找字典
    lookup = {(p.x, p.y): i for i, p in zip(nodes.index, nodes.geometry)}
    edges["u_geom"] = edges["u_geom"].apply(lambda p: set_precision(p, GRID_TOL_M))
    edges["v_geom"] = edges["v_geom"].apply(lambda p: set_precision(p, GRID_TOL_M))
    edges["u"] = edges["u_geom"].apply(lambda pt: lookup.get((pt.x, pt.y)))
    edges["v"] = edges["v_geom"].apply(lambda pt: lookup.get((pt.x, pt.y)))
    edges = edges[edges["u"].notna() & edges["v"].notna()].copy()
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    # 4. 生成图论网络
    edges = edges.drop(columns=["u_geom", "v_geom"])
    edges["key"] = edges.groupby(["u", "v"]).cumcount()
    edges = edges.set_index(["u", "v", "key"])

    Gm = ox.convert.graph_from_gdfs(nodes, edges, graph_attrs={"crs": edges.crs})
    G = Gm.to_undirected(reciprocal=False)

    # 5. 连通性修复
    if nx.number_connected_components(G) > 1:
        components = list(nx.connected_components(G))
        main_comp = max(components, key=len)

        base_nodes = nodes.loc[list(main_comp)]
        base_xy = np.c_[base_nodes.geometry.x, base_nodes.geometry.y]
        tree = KDTree(base_xy)

        for comp in components:
            if comp is main_comp:
                continue
            comp_nodes = nodes.loc[list(comp)]
            comp_xy = np.c_[comp_nodes.geometry.x, comp_nodes.geometry.y]

            dist, idx = tree.query(comp_xy, k=1)
            j = dist.argmin()
            u = comp_nodes.index[j]
            v = base_nodes.index[idx[j][0]]
            length = float(dist[j][0])

            geom = LineString([nodes.loc[u].geometry, nodes.loc[v].geometry])
            attrs = dict(length=length, geometry=geom, speed=30, time_m=round(length/(30*60), 2))
            G.add_edge(u, v, key=0, **attrs)

    # 补齐所有边的长度属性
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" not in data or data["length"] is None:
            data["length"] = float(data["geometry"].length)

    return nx.MultiGraph(G)


# ================= 执行区 =================
if __name__ == "__main__":
    input_shp = r"E:\research4\studyarea_data\Processed_Fixed\edges_fixed_studyarea1.shp"
    
    # 时间段列表：060000-180000，每小时一个
    TIME_SLOTS = [f"{h:02d}0000" for h in range(6, 7)]  # 060000, 070000, ..., 180000
    
    print(f"正在读取拓扑修复后的路网: {input_shp}")
    edges_gdf = gpd.read_file(input_shp)
    
    for time_str in TIME_SLOTS:
        print(f"\n{'='*60}")
        print(f"处理时段: {time_str}")
        print(f"{'='*60}")
        
        output_graphml = fr"E:\research4\studyarea_data\fixed_roadmap\fixed_road_graph_f{time_str}.graphml"
        os.makedirs(os.path.dirname(output_graphml), exist_ok=True)
        
        G = build_graph_from_fixed_edges(edges_gdf, time_str=time_str)
        
        print(f"连通分量数：{nx.number_connected_components(G)}")
        print(f"节点数: {len(G.nodes)}, 线段数: {len(G.edges)}")
        
        ox.save_graphml(G, filepath=output_graphml)
        print(f"已保存: {output_graphml}")
    
    print(f"\n全部完成！共生成 {len(TIME_SLOTS)} 个时段的路网图")
