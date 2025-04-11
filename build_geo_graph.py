import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math


def haversine(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半径 (单位：千米)
    R = 6371.0
    distance = R * c
    return distance


def build_global_geo_adjacency(data_node_feats):
    # 读取POI数据
    nodes_df = pd.read_csv(data_node_feats)

    # POI id 到 index 的映射
    poi_id2idx_dict = {poi_id: idx for idx, poi_id in enumerate(nodes_df['node_name/poi_id'])}
    poi_ids = list(poi_id2idx_dict.keys())

    # 获取所有POI的经纬度
    poi_coords = {}
    for _, row in nodes_df.iterrows():
        poi_id = row['node_name/poi_id']
        latitude = row['latitude']
        longitude = row['longitude']
        poi_coords[poi_id] = (latitude, longitude)

    # 初始化POI-POI地理距离矩阵
    num_pois = len(poi_id2idx_dict)
    geo_distance_matrix = np.zeros((num_pois, num_pois), dtype=float)

    # 填充地理距离矩阵
    for poi_id_1 in tqdm(poi_coords.keys(), desc="Building POI-POI geospatial distance matrix"):
        for poi_id_2 in poi_coords.keys():
            if poi_id_1 != poi_id_2:
                coord_1 = poi_coords[poi_id_1]
                coord_2 = poi_coords[poi_id_2]
                # 使用Haversine公式计算POI之间的地理距离
                distance = haversine(coord_1[0], coord_1[1], coord_2[0], coord_2[1])
                idx_1 = poi_id2idx_dict[poi_id_1]
                idx_2 = poi_id2idx_dict[poi_id_2]
                geo_distance_matrix[idx_1, idx_2] = distance

    # 进行平方根缩放到[0, 1]
    max_distance = np.max(geo_distance_matrix)
    min_distance = np.min(geo_distance_matrix)

    # 平方根缩放，避免取负值
    if max_distance != min_distance:
        geo_distance_matrix = (geo_distance_matrix - min_distance) / (
                max_distance - min_distance)
    else:
        geo_distance_matrix = np.zeros_like(geo_distance_matrix)

    return geo_distance_matrix, poi_id2idx_dict


if __name__ == '__main__':
    # 定义地点列表
    locations = ['SIN', 'CAL', 'NYC', 'PHO']

    # 遍历每个地点并处理
    for location in locations:
        print(f'Processing {location} data...')

        # 定义数据路径
        dst_dir = os.path.join('../dataset', location)

        data_node_feats = os.path.join(dst_dir, f'graph_X.csv')

        # 构建 geo 邻接矩阵
        print(f'Building global POI geo adjacency matrix for {location} -----------------------------------')
        adjacency_matrix, poi_id2idx_dict = build_global_geo_adjacency(data_node_feats)

        # 保存邻接矩阵到 CSV 文件
        pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dst_dir, f'graph_geo.csv'), index=False, header=False)

        print(f'{location} data processing completed.\n')
