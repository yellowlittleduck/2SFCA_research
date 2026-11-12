import geopandas as gpd
from shapely.ops import snap
from shapely.geometry import Point, MultiPoint
from shapely.strtree import STRtree
from tqdm import tqdm
import os

def process_hangzhou_road_topology(input_path, output_folder, snap_tolerance=1.5):
    """
    参考 test.ipynb 逻辑，修复拓扑并导出 edges 和 nodes 两个 SHP 文件 
    """
    print(f"1. 正在读取原始路网数据: {input_path}")
    edges = gpd.read_file(input_path)
    
    print("2. 正在拆分多部件几何体 (Explode)...")
    # 参考 ipynb，通过 explode 确保所有几何体均为 LineString 
    edges = edges.explode(index_parts=False, ignore_index=True)
    edges = edges[edges.geometry.type == 'LineString'].reset_index(drop=True)

    print("3. 正在提取节点并构建空间索引...")
    # 提取每条线的起点和终点作为潜在节点 
    nodes_list = []
    for idx, row in edges.iterrows():
        geom = row.geometry
        nodes_list.append({'geometry': Point(geom.coords[0])})
        nodes_list.append({'geometry': Point(geom.coords[-1])})
    
    nodes_gdf = gpd.GeoDataFrame(nodes_list, crs=edges.crs)
    # 移除重复位置的点，得到纯净的节点层
    nodes_gdf = nodes_gdf.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    nodes_gdf['node_id'] = nodes_gdf.index

    # 构建空间索引树以加速捕捉 
    node_geoms = nodes_gdf.geometry.values
    tree = STRtree(node_geoms)
    
    def fast_snap(line, tolerance):
        search_area = line.buffer(tolerance)
        candidate_indices = tree.query(search_area)
        if len(candidate_indices) == 0:
            return line
        nearby_nodes = MultiPoint(node_geoms[candidate_indices])
        return snap(line, nearby_nodes, tolerance)

    print(f"4. 正在执行空间捕捉 (容差: {snap_tolerance}米)...")
    tqdm.pandas(desc="路网捕捉进度")
    edges['geometry'] = edges['geometry'].progress_apply(lambda x: fast_snap(x, snap_tolerance))
    
    # 5. 导出结果 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"5. 正在导出文件至: {output_folder}")
    # 导出修复后的边文件
    edges.to_file(os.path.join(output_folder, "edges_fixed.shp"))
    # 导出生成的节点文件
    nodes_gdf.to_file(os.path.join(output_folder, "nodes_fixed.shp"))
    
    print("✅ 导出完成！已生成 edges_fixed.shp 和 nodes_fixed.shp ")

# ================= 运行区 =================
if __name__ == "__main__":
    RAW_PATH = r"E:\research4\studyarea_data\origin_roadmap_UTM.shp"
    # 这里填保存文件的文件夹路径
    SAVE_FOLDER = r"E:\research4\studyarea_data\Processed_Fixed"
    
    process_hangzhou_road_topology(RAW_PATH, SAVE_FOLDER, snap_tolerance=1.5)