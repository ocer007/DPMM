import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, train_df, user_id2idx_dict, poi_id2idx_dict, args):
        self.df = train_df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        self.user_id2idx_dict = user_id2idx_dict
        self.poi_id2idx_dict = poi_id2idx_dict
        self.args = args

        self.max_seq_len = 0  # 用于存储最长序列的长度

        for traj_id in tqdm(set(train_df['traj_id'].tolist())):
            traj_df = train_df[train_df['traj_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = [self.poi_id2idx_dict[each] for each in poi_ids]
            time_feature = traj_df[self.args.time_feature].to_list()

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            if len(input_seq) < self.args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

            # 更新最大序列长度
            self.max_seq_len = max(self.max_seq_len, len(input_seq))

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    def get_max_sequence_length(self):
        # 返回最长序列长度
        return self.max_seq_len



class TrajectoryDatasetVal(Dataset):
    def __init__(self, df, user_id2idx_dict, poi_id2idx_dict, args):
        self.df = df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        self.user_id2idx_dict = user_id2idx_dict
        self.poi_id2idx_dict = poi_id2idx_dict
        self.args = args

        self.max_seq_len = 0  # 用于存储最长序列的长度

        for traj_id in tqdm(set(df['traj_id'].tolist())):
            user_id = traj_id.split('_')[0]

            if user_id not in self.user_id2idx_dict:
                continue

            traj_df = df[df['traj_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = [self.poi_id2idx_dict[each] for each in poi_ids if each in self.poi_id2idx_dict]
            time_feature = traj_df[self.args.time_feature].to_list()

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            if len(input_seq) < self.args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

            # 更新最大序列长度
            self.max_seq_len = max(self.max_seq_len, len(input_seq))

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    def get_max_sequence_length(self):
        # 返回最长序列长度
        return self.max_seq_len
