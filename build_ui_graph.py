import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_global_UI_adjacency(train_df, data_node_feats):
    # POI id to index
    nodes_df = pd.read_csv(data_node_feats)
    poi_id2idx_dict = {poi_id: idx for idx, poi_id in enumerate(nodes_df['node_name/poi_id'])}
    poi_ids = list(poi_id2idx_dict.keys())

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # 初始化用户-POI 邻接矩阵
    num_users = len(user_id2idx_dict)
    num_pois = len(poi_id2idx_dict)
    adjacency_matrix = np.zeros((num_users, num_pois), dtype=int)

    # 填充邻接矩阵
    for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc="Building adjacency matrix"):
        user_id = str(row['user_id'])
        poi_id = row['POI_id']

        if user_id in user_id2idx_dict and poi_id in poi_id2idx_dict:
            user_idx = user_id2idx_dict[user_id]
            poi_idx = poi_id2idx_dict[poi_id]
            adjacency_matrix[user_idx, poi_idx] += 1  # 增加交互次数

    return adjacency_matrix, user_id2idx_dict, poi_id2idx_dict


if __name__ == '__main__':
    # 定义地点列表
    locations = ['SIN','CAL', 'NYC', 'PHO']

    # 遍历每个地点并处理
    for location in locations:
        print(f'Processing {location} data...')

        # 定义数据路径
        dst_dir = os.path.join('../dataset', location)

        # 读取数据
        train_df = pd.read_csv(os.path.join(dst_dir, f'{location}_train.csv'))
        data_node_feats = os.path.join(dst_dir, f'graph_X.csv')

        # 构建 U-I 邻接矩阵
        print(f'Building global POI checkin adjacency matrix for {location} -----------------------------------')
        adjacency_matrix, user_id2idx_dict, poi_id2idx_dict = build_global_UI_adjacency(train_df, data_node_feats)

        # 保存邻接矩阵到 CSV 文件
        pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dst_dir, f'graph_UI.csv'), index=False, header=False)

        print(f'{location} data processing completed.\n')
