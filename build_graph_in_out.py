import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

def build_global_POI_checkin_graphs(df, exclude_user=None):
    incoming_graph = nx.DiGraph()
    outgoing_graph = nx.DiGraph()

    users = list(set(df['user_id'].to_list()))
    if exclude_user in users:
        users.remove(exclude_user)

    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add nodes (POI)
        for i, row in user_df.iterrows():
            node = row['POI_id']
            if node not in incoming_graph.nodes():
                incoming_graph.add_node(row['POI_id'],
                                        checkin_cnt=1,
                                        poi_catid=row['POI_catid'],
                                        poi_catid_code=row['POI_catid_code'],
                                        poi_catname=row['POI_catname'],
                                        latitude=row['latitude'],
                                        longitude=row['longitude'])
                outgoing_graph.add_node(row['POI_id'],
                                        checkin_cnt=1,
                                        poi_catid=row['POI_catid'],
                                        poi_catid_code=row['POI_catid_code'],
                                        poi_catname=row['POI_catname'],
                                        latitude=row['latitude'],
                                        longitude=row['longitude'])
            else:
                incoming_graph.nodes[node]['checkin_cnt'] += 1
                outgoing_graph.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['traj_id']

            # No edge for the beginning of the sequence or different trajectory
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges to the outgoing graph
            if outgoing_graph.has_edge(previous_poi_id, poi_id):
                outgoing_graph.edges[previous_poi_id, poi_id]['weight'] += 1
            else:
                outgoing_graph.add_edge(previous_poi_id, poi_id, weight=1)

            # Add edges to the incoming graph
            if incoming_graph.has_edge(poi_id, previous_poi_id):
                incoming_graph.edges[poi_id, previous_poi_id]['weight'] += 1
            else:
                incoming_graph.add_edge(poi_id, previous_poi_id, weight=1)

            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return incoming_graph, outgoing_graph

def save_graph_to_csv(G, dst_dir, prefix):
    # Save adjacency matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, f'{prefix}_graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, f'{prefix}_graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},{poi_catname},'
                  f'{latitude},{longitude}', file=f)

def save_graph_to_pickle(G, dst_dir, prefix):
    pickle.dump(G, open(os.path.join(dst_dir, f'{prefix}_graph.pkl'), 'wb'))

def save_graph_edgelist(G, dst_dir, prefix):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, f'{prefix}_graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, f'{prefix}_graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)

if __name__ == '__main__':
    # Define location list
    locations = ['CAL', 'NYC', 'PHO', 'SIN']

    # Process each location
    for location in locations:
        print(f'Processing {location} data...')

        # Define data path
        dst_dir = os.path.join('../dataset', location)

        # Read data
        train_df = pd.read_csv(os.path.join(dst_dir, f'{location}_train.csv'))

        # Build POI check-in graphs
        print(f'Building global POI checkin graphs for {location} -----------------------------------')
        incoming_graph, outgoing_graph = build_global_POI_checkin_graphs(train_df)

        # Save graphs to disk
        # save_graph_to_pickle(incoming_graph, dst_dir, prefix='incoming')
        save_graph_to_csv(incoming_graph, dst_dir, prefix='in')
        # save_graph_edgelist(incoming_graph, dst_dir, prefix='incoming')

        # save_graph_to_pickle(outgoing_graph, dst_dir, prefix='outgoing')
        save_graph_to_csv(outgoing_graph, dst_dir, prefix='out')
        # save_graph_edgelist(outgoing_graph, dst_dir, prefix='outgoing')

        print(f'{location} data processing completed.\n')
