# DPMM

## Requirements

```
torch==2.0.0
torch-summary==1.4.5
numpy==1.24.4
pandas==1.1.5
prettytable==2.0.0
matplotlib==3.3.4
scipy==1.6.1
tqdm==4.58.0
data==0.4
```

## Train

For PHO dataset:

```
python train.py --data-train ../dataset/PHO/PHO_train.csv --data-val ../dataset/PHO/PHO_val.csv \
                --data-adj-mtx ../dataset/PHO/graph_A.csv --data-node-feats ../dataset/PHO/graph_X.csv \
                --data-adj-mtx-in ../dataset/PHO/in_graph_A.csv  --data-adj-mtx-out ../dataset/PHO/out_graph_A.csv \
                --data-UI-mtx ../dataset/PHO/graph_UI.csv \
                --data-geo-mtx ../dataset/PHO/graph_geo.csv \
                --poi-image-embedding ../dataset/PHO/PHO_image_encoded.json \
                --poi-comment-embedding ../dataset/PHO/PHO_POI_comments_encoded.json \
                --poi-meta-embedding ../dataset/PHO/PHO_POI_meta_encoded_combined.json \
                --device cuda:0 \
                --name train_v9-0.6 \
                --project runs/PHO \
                --alpha 0.6
```