import geopandas as gpd
import os

def apply_solid_buffer(input_shp, output_shp):
    """
    回归初心：基于道路等级的实心缓冲区 (Solid Buffer)
    """
    print(f"1. 正在读取拓扑修复后的路网数据: {input_shp}")
    edges = gpd.read_file(input_shp)
    
    # # 确保数据在平面坐标系 (杭州 EPSG:32651) 下，单位为米
    # if edges.crs is None or edges.crs.to_epsg() != 32651:
    #     print("   => 正在投影至 EPSG:32651...")
    #     edges = edges.to_crs(epsg=32651)
        
    print("2. 正在映射中国道路标准缓冲半径...")
    # 核心字典：基于 CJJ 37-2012 与 Tang et al. 2025 标准
    buffer_radius_dict = {
        'primary': 10.0,
        'primary_link': 10.0,
        'secondary': 8.0,
        'secondary_link': 8.0,
        'tertiary': 6.0,
        'tertiary_link': 6.0,
        'residential': 4.0,
        'unclassified': 4.0,
        'pedestrian': 3.0,
        'living_street': 3.0,
        'service': 3.0,
        'footway': 1.5,
        'cycleway': 1.5,
        'path': 1.5,
        'steps': 1.5,
        'track': 2.0,
        'track_grade2': 2.0,
        'track_grade4': 2.0,
        'unknown': 2.0
    }
    
    # 根据 'fclass' (或你的道路等级列名) 映射宽度，如果遇到未知的给 2.0 米
    # 注意：请检查你的 SHP 文件中道路类型对应的列名是 'fclass' 还是 'highway'
    road_type_col = 'fclass' if 'fclass' in edges.columns else 'highway'
    edges['buf_radius'] = edges[road_type_col].map(buffer_radius_dict).fillna(2.0)
    
    print("3. 正在生成实心 Buffer (使用 cap_style=2 保证路口垂直切面)...")
    # cap_style=2 (平头) 极其关键，它能防止路口的 Buffer 变成半圆形重叠，导致面积重复计算
    edges['geometry'] = edges.apply(
        lambda row: row['geometry'].buffer(row['buf_radius'], cap_style=2), 
        axis=1
    )
    
    print(f"4. 正在保存生成的实心面数据至: {output_shp}")
    # 获取输出文件夹路径并确保其存在
    out_folder = os.path.dirname(output_shp)
    if not os.path.exists(out_folder) and out_folder != '':
        os.makedirs(out_folder)
        
    # 保存结果 (保持 EPSG:32651，方便你直接去扣栅格阴影数据)
    edges.to_file(output_shp)
    print("✅ 任务完成！实心路网 Buffer 已生成。")

# ================= 运行区 =================
if __name__ == "__main__":
    # 输入文件：你上一轮刚刚做完拓扑修复的 edges_fixed.shp
    INPUT_PATH = r"E:\research4\studyarea_data\Processed_Fixed\edges_fixed_studyarea.shp"
    
    # 输出文件：准备用来计算阴影覆盖率的 Polygon 面文件
    OUTPUT_PATH = r"E:\research4\studyarea_data\Processed_Fixed\edges_buffered_studyarea.shp"
    
    apply_solid_buffer(INPUT_PATH, OUTPUT_PATH)