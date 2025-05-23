"""Parsing the parameters."""
import argparse

import torch

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--seed',
                        type=int,
                        default=210,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')

    # 添加 POI 多模态数据路径
    parser.add_argument('--poi-multi-id',
                        type=str,
                        default='../dataset/PHO/PHO_POI_id_mapping.pkl',
                        help='Path to POI mutil id file')
    parser.add_argument('--poi-image-embedding',
                        type=str,
                        default='../dataset/PHO/PHO_image_encoded.json',
                        help='Path to POI image embedding file')
    parser.add_argument('--poi-comment-embedding',
                        type=str,
                        default='../dataset/PHO/PHO_POI_comments_encoded.json',
                        help='Path to POI comment embedding file')
    parser.add_argument('--poi-meta-embedding',
                        type=str,
                        default='../dataset/PHO/PHO_POI_meta_encoded_combined.json',
                        help='Path to POI meta embedding file')

    # multi推荐占比
    parser.add_argument('--alpha',
                        type=float,
                        default=0.5,
                        help='multi poi rec percent')

    parser.add_argument('--beta',
                        type=float,
                        default=0.5,
                        help='ID poi rec percent')
    parser.add_argument('--lamda',
                        type=float,
                        default=0.5,
                        help='MM present')

    parser.add_argument('--image-weight',
                        type=float,
                        default=0.5)
    parser.add_argument('--comment-weight',
                        type=float,
                        default=1.5)
    parser.add_argument('--meta-weight',
                        type=float,
                        default=1.5)

    # Data
    parser.add_argument('--data-geo-mtx',
                        type=str,
                        default='../dataset/PHO/graph_geo.csv',
                        help='geo Graph adjacent path')
    parser.add_argument('--data-UI-mtx',
                        type=str,
                        default='../dataset/PHO/graph_UI.csv',
                        help='UI Graph adjacent path')
    parser.add_argument('--data-adj-mtx',
                        type=str,
                        default='../dataset/PHO/graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-adj-mtx-in',
                        type=str,
                        default='../dataset/PHO/in_graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-adj-mtx-out',
                        type=str,
                        default='../dataset/PHO/out_graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-node-feats',
                        type=str,
                        default='../dataset/PHO/graph_X.csv',
                        help='Graph node features path')
    parser.add_argument('--data-train',
                        type=str,
                        default='../dataset/PHO/PHO_train.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='../dataset/PHO/PHO_val.csv',
                        help='Validation data path')
    parser.add_argument('--short-traj-thres',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')
    parser.add_argument('--time-feature',
                        type=str,
                        default='norm_in_day_time',
                        help='The name of time feature in the data')

    # Model hyper-parameters
    parser.add_argument('--NLP-embedding-dim',
                        type=int,
                        default=768,
                        help='The dimension of BERT')
    parser.add_argument('--multi-reshape-dim',
                        type=int,
                        default=128,
                        help='The dimension of BERT')
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='SPHOe factor for the time loss term')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=150,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/debug2',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')

    return parser.parse_args()
