import glob
import math
import os
import random
import re
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh
from sentence_transformers import SentenceTransformer


def multi_embedding_reshape(args, model, embedding):
    input_data = []
    poiid_list = []  # 用于存储poiid
    for key, value in embedding.items():
        input_data.append(value)
        poiid_list.append(key)  # 保存对应的poiid
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device=args.device)
    output_tensor = model(input_tensor)
    output_dict = {poiid: output_tensor[i].detach().cpu().numpy() for i, poiid in enumerate(poiid_list)}
    return output_dict


def input_traj_to_embeddings(sample, poi_embeddings, user_embed_model, time_embed_model, cat_embed_model,
                             embed_fuse_model1, embed_fuse_model2, user_id2idx_dict, poi_idx2cat_idx_dict, device,
                             args):
    # Parse sample
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]
    input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

    # User to embedding
    user_id = traj_id.split('_')[0]
    user_idx = user_id2idx_dict[user_id]
    input = torch.LongTensor([user_idx]).to(device=device)
    user_embedding = user_embed_model(input)
    user_embedding = torch.squeeze(user_embedding)

    # POI to embedding and fuse embeddings
    input_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]]
        poi_embedding = torch.squeeze(poi_embedding).to(device=device)

        # Time to vector
        time_embedding = time_embed_model(
            torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=device))
        time_embedding = torch.squeeze(time_embedding).to(device=device)

        # Category to embedding
        cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=device)
        cat_embedding = cat_embed_model(cat_idx)
        cat_embedding = torch.squeeze(cat_embedding)

        # Fuse user+poi embeds
        fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
        # Fuse cat+time embeds
        fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

        # Concat time, cat after user+poi
        concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

        # Save final embed
        input_seq_embed.append(concat_embedding)

    return input_seq_embed


def input_traj_to_multi_embeddings(sample, poi_embeddings, user_embed_model, time_embed_model, cat_embed_model,
                                   embed_fuse_model1, embed_fuse_model2, user_id2idx_dict, poi_idx2cat_idx_dict, device,
                                   args):
    # Parse sample
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]
    input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

    # User to embedding
    user_id = traj_id.split('_')[0]
    user_idx = user_id2idx_dict[user_id]
    input = torch.LongTensor([user_idx]).to(device=device)
    user_embedding = user_embed_model(input)
    user_embedding = torch.squeeze(user_embedding)

    # POI to embedding and fuse embeddings
    input_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]]
        poi_embedding = torch.squeeze(poi_embedding).to(device=device)

        # Time to vector
        time_embedding = time_embed_model(
            torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=device))
        time_embedding = torch.squeeze(time_embedding).to(device=device)

        # Category to embedding
        cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=device)
        cat_embedding = cat_embed_model(cat_idx)
        cat_embedding = torch.squeeze(cat_embedding)

        # Fuse user+poi embeds
        fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
        fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

        # Concat time, cat after user+poi
        concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

        # Save final embed
        input_seq_embed.append(concat_embedding)

    return input_seq_embed


def adjust_pred_prob_by_graph(y_pred_poi, node_attn_model, X, A, batch_input_seqs, batch_seq_lens):
    y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
    attn_map = node_attn_model(X, A)

    for i in range(len(batch_seq_lens)):
        traj_i_input = batch_input_seqs[i]  # list of input check-in pois
        for j in range(len(traj_i_input)):
            y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

    return y_pred_poi_adjusted


def load_multimodal_data(filename, embedding_dim=768):
    # 生成缓存文件路径
    filename_saved = filename.replace('.json', '.pkl')

    # 如果缓存文件存在，加载缓存文件
    if os.path.exists(filename_saved):
        with open(filename_saved, 'rb') as f:
            data = pickle.load(f)
        return data

    # 加载JSON文件并处理为字典形式
    with open(filename, 'r') as f:
        data = f.readlines()
        data = [eval(line) for line in data]
        data = {k: v for d in data for k, v in d.items()}

        # 设置填充值
        padding = [0.0] * embedding_dim
        for k, v in data.items():
            # 如果嵌入长度不符合预期，填充为默认的0向量
            if len(v) != embedding_dim:
                data[k] = padding

    # 将处理后的数据保存为缓存文件
    with open(filename_saved, 'wb') as f:
        pickle.dump(data, f)

    return data


import numpy as np


def compute_average_embedding_v1(poi_id2idx_dict, nodes_df, POI_image_embedding,
                                 POI_comment_embedding, POI_meta_embedding,
                                 device, embedding_dim=768):
    POI_mutil_embedding = {}
    padding = np.zeros(embedding_dim)  # 用于嵌入缺失的填充值

    model_sentence = SentenceTransformer(
        '/home/lgl/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2', device=device)

    for key in poi_id2idx_dict:
        try:
            # 获取各个嵌入，如果不存在则用填充值
            image_embed = np.array(POI_image_embedding.get(key, padding))
            comment_embed = np.array(POI_comment_embedding.get(key, padding))
            meta_embed = np.array(POI_meta_embedding.get(key, padding))

            # 如果三个嵌入都不存在，尝试从 DataFrame 获取 poi_catname
            if np.array_equal(image_embed, padding) and np.array_equal(comment_embed, padding) and np.array_equal(
                    meta_embed, padding):
                category = nodes_df.loc[nodes_df['node_name/poi_id'] == key, 'poi_catname']
                if not category.empty:
                    meta_embed = model_sentence.encode(category.values[0])

        except Exception as e:
            print(f"Error processing key: {key}, Error: {e}")
            continue  # 遇到错误继续处理下一个 key

        # 检查可用的嵌入数量，保存不是填充值的值
        available_embeddings = [emb for emb in [image_embed, comment_embed, meta_embed] if
                                not np.array_equal(emb, padding)]

        # 计算平均嵌入
        POI_mutil_embedding[poi_id2idx_dict[key]] = np.mean(available_embeddings,
                                                            axis=0) if available_embeddings else padding

    return POI_mutil_embedding


def compute_average_embedding_v3(model_sentence, poi_id2idx_dict, nodes_df, multi_reshape_model, POI_image_embedding,
                                 POI_comment_embedding, POI_meta_embedding,
                                 device, embedding_dim=768):
    POI_mutil_embedding = {}
    padding = np.zeros(embedding_dim)  # 用于嵌入缺失的填充值

    for key in poi_id2idx_dict:
        try:
            # 获取各个嵌入，如果不存在则用填充值
            image_embed = np.array(POI_image_embedding.get(key, padding))
            comment_embed = np.array(POI_comment_embedding.get(key, padding))
            meta_embed = np.array(POI_meta_embedding.get(key, padding))

            # 如果三个嵌入都不存在，尝试从 DataFrame 获取 poi_catname
            if np.array_equal(image_embed, padding) and np.array_equal(comment_embed, padding) and np.array_equal(
                    meta_embed, padding):
                category = nodes_df.loc[nodes_df['node_name/poi_id'] == key, 'poi_catname']
                if not category.empty:
                    meta_BERT_embed = model_sentence.encode(category.values[0])
                    meta_BERT_embed = torch.tensor(meta_BERT_embed, dtype=torch.float32).to(device=device)
                    meta_embed = multi_reshape_model(meta_BERT_embed)
                    meta_embed = np.array(meta_embed.detach().cpu().numpy())

        except Exception as e:
            print(f"Error processing key: {key}, Error: {e}")
            continue  # 遇到错误继续处理下一个 key

        # 检查可用的嵌入数量，保存不是填充值的值
        available_embeddings = [emb for emb in [image_embed, comment_embed, meta_embed] if
                                not np.array_equal(emb, padding)]

        # 计算平均嵌入
        POI_mutil_embedding[poi_id2idx_dict[key]] = np.mean(available_embeddings,
                                                            axis=0) if available_embeddings else padding

    return POI_mutil_embedding


def compute_average_embedding_v4(poi_id2idx_dict, nodes_df, POI_image_embedding,
                                 POI_comment_embedding, POI_meta_embedding,
                                 device, embedding_dim, image_weight, comment_weight, meta_weight):
    POI_mutil_embedding = {}
    padding = np.zeros(embedding_dim)  # 用于嵌入缺失的填充值

    model_sentence = SentenceTransformer(
        '/home/lgl/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2', device=device)

    for key in poi_id2idx_dict:
        try:
            # 获取各个嵌入，如果不存在则用填充值
            image_embed = np.array(POI_image_embedding.get(key, padding)) * image_weight
            comment_embed = np.array(POI_comment_embedding.get(key, padding)) * comment_weight
            meta_embed = np.array(POI_meta_embedding.get(key, padding)) * meta_weight

            # 如果三个嵌入都不存在，尝试从 DataFrame 获取 poi_catname
            if np.array_equal(image_embed, padding * image_weight) and np.array_equal(comment_embed,
                                                                                      padding * 1.5) and np.array_equal(
                meta_embed, padding * 1.5):
                category = nodes_df.loc[nodes_df['node_name/poi_id'] == key, 'poi_catname']
                if not category.empty:
                    meta_embed = model_sentence.encode(category.values[0]) * meta_weight

        except Exception as e:
            print(f"Error processing key: {key}, Error: {e}")
            continue  # 遇到错误继续处理下一个 key

        # 检查可用的嵌入数量，保存不是填充值的值
        available_embeddings = [emb for emb in [image_embed, comment_embed, meta_embed] if
                                not np.array_equal(emb, padding)]

        # 计算平均嵌入
        POI_mutil_embedding[poi_id2idx_dict[key]] = np.mean(available_embeddings,
                                                            axis=0) if available_embeddings else padding

    return POI_mutil_embedding


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))
