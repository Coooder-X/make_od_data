import json
import os
import pickle
from datetime import datetime
import time
import threading

# from cdlib import algorithms
import igraph as ig
import numpy as np
import pandas as pd
import torch

from SpatialRegionTools import get_cell_id_center_coord_dict
from args import args
from gcc.graph_convolutional_clustering.gcc.run import run
# import torch
#
# import utils
from graph_process.Point import Point
# from cal_od import exp_od_pair_set, get_od_filter_by_day_and_hour, get_od_hot_cell, encode_od_point
# import od_pair_process
# from data_process.OD_area_graph import build_od_graph, fuse_fake_edge_into_linegraph, \
#     get_line_graph_by_selected_cluster
# from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS
# from data_process.od_pair_process import get_trips_by_ids, get_trj_ids_by_force_node, \
#     get_odpair_space_similarity
# from data_process.spatial_grid_utils import get_region, get_od_points_filter_by_region, divide_od_into_grid
# from gcc.graph_convolutional_clustering.gcc.run import run, draw_cluster_in_trj_view, draw_cluster_in_trj_view_new
from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix, Graph, networkx2igraph, igraph2networkx
# from t2vec import args
# from t2vec_graph import run_model2, get_cluster_by_trj_feature
import networkx as nx

from od_graph_process import get_line_graph_by_selected_cluster
from od_pair_process import get_od_points_filter_by_day_and_hour, get_od_filter_by_day_and_hour
from spatial_grid_utils import get_region, decode_od
from t2vec import run_model2
from visualization import vis_community

# G = nx.karate_club_graph()
# g = ig.Graph(directed=True)
# g = g.from_networkx(G)
# G = igraph2networkx(g, nx.MultiDiGraph)
# print('G -----<>', G)
# gl = g.linegraph()
# print('line graph ======>', gl)
# print('edges ======>', list(gl.es))
# com = g.community_edge_betweenness(clusters=3, directed=True, weights=None)
# print('com ===========>', com.as_clustering())


exp3_log_name = 'exp5_log'
exp3_log = []


def get_trj_feats(gps_trips, best_model):
    with open("../region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
    trj_feats = run_model2(args, gps_trips, best_model, trj_region)
    # print('gps_trips', gps_trips)
    # print('trj_feats', trj_feats)
    return trj_feats


def CON(G, cluster_id, node_name_cluster_dict):
    start = datetime.now()
    m = len(G.edges())
    fz = 0
    for edge in G.edges():
        u, v = edge[0], edge[1]
        u_name, v_name = u, v
        if use_line_graph:
            u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        # print(u_name, v_name)
        # print(node_name_cluster_dict.keys())
        # u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        if (node_name_cluster_dict[u_name] == cluster_id and node_name_cluster_dict[v_name] != cluster_id) or \
                (node_name_cluster_dict[u_name] != cluster_id and node_name_cluster_dict[v_name] == cluster_id):
            fz += 1
    vol_C = vol(G, cluster_id, node_name_cluster_dict)
    print(f'vol_C={vol_C}({cluster_id})')
    fm = fz + vol_C
    # fm = min(vol_C, m - vol_C)
    if fm == 0 or fz == 0:
        return -1
    end = datetime.now()
    print('用时', end - start)
    print(f'分子={fz}， 分母={fm}')
    # res = fz / (fz + vol_C + 0.01)
    # print(f'CON=({res})')
    res = fz / fm
    return res


def vol(G, cluster_id, node_name_cluster_dict):
    res = 0
    # print(f'vol ==== G.nodes() = {G.nodes()}')
    print(node_name_cluster_dict.keys())
    for node in G.nodes():
        node_name = node
        if use_line_graph:
            node_name = f'{node[0]}_{node[1]}'
        if node_name in node_name_cluster_dict and node_name_cluster_dict[node_name] == cluster_id:
            res += G.degree(node)
    return res


def avg_CON(G, cluster_point_dict, node_name_cluster_dict, use_igraph):
    if use_igraph is True:
        G = igraph2networkx(G, nx.MultiDiGraph)
    avg = 0.0
    # print(len(cluster_point_dict.keys()))
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        # if len(cluster_point_dict[cluster_id]) > 5:
        cur_con = CON(G, cluster_id, node_name_cluster_dict)
        if cur_con == -1:
            continue
        ok_cluster_num += 1
        avg += cur_con
        print(f'cluster: {cluster_id} cur_con = {cur_con}')

    avg /= ok_cluster_num
    exp3_log.append(f'cluster_num {len(cluster_point_dict.keys())} avg Con = {avg}')
    return avg


def get_ok_cluster_num(cluster_point_dict):
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        if len(cluster_point_dict[cluster_id]) > 5:
            ok_cluster_num += 1
    return ok_cluster_num


consider_edge_weight = True
use_line_graph = False
use_igraph = False


def get_origin_graph_by_selected_cluster(out_adj_dict, data_id):
    cluster_list = []  # 存储所有 Point 类型的 簇，作为 graph 的节点集
    cid_point_dict = {}  # 簇id 到 Point 类型的簇 的映射
    point_cid_dict = {}  # Point 类型的簇 到 簇id 的映射
    cluster_ids = set()
    for key in out_adj_dict:
        cluster_ids.add(key)
        for v in out_adj_dict[key]:
            cluster_ids.add(v)

    for cid in cluster_ids:
        point = Point(name=cid, nodeId=cid, infoObj={}, feature={})
        cluster_list.append(point)
        cid_point_dict[cid] = point
        point_cid_dict[point] = cid

    adj_point_dict = {}  # 根据 filtered_adj_dict 得出的等价的邻接表，索引是 Point 类型的簇
    for cid in out_adj_dict:
        point = cid_point_dict[cid]
        if point not in adj_point_dict:
            adj_point_dict[point] = []
        for to_cid in out_adj_dict[cid]:
            adj_point_dict[point].append(cid_point_dict[to_cid])

    od_flow_dict = {}
    with open(f"../data/selected_od_trj_dict_{data_id}.pkl", 'rb') as file:
        selected_od_trj_dict = pickle.loads(file.read())
    for key in selected_od_trj_dict:
        o, d = decode_od(key)
        if key == '55_66' or key == '66_55' or key == '53_35':
            continue
        od_flow_dict[(o, d)] = len(selected_od_trj_dict[key])

    g = Graph()
    for cluster in cluster_list:
        g.addVertex(cluster)
    #   边权值可以后续改成簇之间的 od 对数量，暂时默认为 1
    for point in adj_point_dict:
        edge = []
        u = point_cid_dict[point]
        for to_point in adj_point_dict[point]:
            v = point_cid_dict[to_point]
            if consider_edge_weight is True and (u, v) in od_flow_dict:
                edge.append([to_point, od_flow_dict[(u, v)]])
            else:
                edge.append([to_point, 1])
        g.addDirectLine(point, edge)
    return g, out_adj_dict


# month = 5
# # start_day, end_day = 11, 12
# start_day, end_day = 12, 14
# # start_hour, end_hour = 18, 20
# start_hour, end_hour = 8, 10


def get_grid_split(data_id):
    out_adj_table = {}
    with open(f"../json_data/od_graph_{data_id}.json") as json_g:
        # 读取配置文件
        od_list = json.load(json_g)
    # od_list = [[od_pair['start'], od_pair['end'], ] for od_pair in od_list]
    for od_pair in od_list:
        if [od_pair['start'], od_pair['end']] == [55, 65] or [od_pair['start'], od_pair['end']] == [65, 55] \
                or [od_pair['start'], od_pair['end']] == [53, 35]:
            continue
        if od_pair['start'] not in out_adj_table:
            out_adj_table[od_pair['start']] = set()
        out_adj_table[od_pair['start']].add(od_pair['end'])
    return out_adj_table


def get_line_graph(region, trj_region, out_adj_table, data_id):
    with_space_dist = False
    # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
    used_od_cells = set([x for x in range(100)])
    tmp = {}
    for start in out_adj_table:
        if start in used_od_cells:
            t = out_adj_table[start]
            t = list(set(t).intersection(used_od_cells))
            tmp[start] = t
    out_adj_table = tmp
    cid_center_coord_dict = get_cell_id_center_coord_dict(region)
    # selected_cluster_ids = list(cid_center_coord_dict.keys())
    selected_cluster_ids = set(cid_center_coord_dict.keys())
    selected_cluster_ids = list(selected_cluster_ids.intersection(used_od_cells))
    if use_line_graph is True:
        g, filtered_adj_dict = get_origin_graph_by_selected_cluster(out_adj_table, data_id)
        force_nodes, force_edges, line_graph_filtered_adj_dict, lg = get_line_graph_by_selected_cluster(
            selected_cluster_ids, selected_cluster_ids, out_adj_table)
        if use_igraph is True:
            g = g.G
            g_tmp = ig.Graph(directed=True)
            g_tmp = g_tmp.from_networkx(g)
            g = g_tmp.linegraph()
        else:
            # g.drawLineGraph()
            g = lg
        # for n in g.nodes:
        #     g_tmp.add_vertex(n)
        # # print(g)
        # for e in g.edges:
        # #     g_tmp.add_edge(ig.Edge(source=e[0], target=e[1]))
        #     g_tmp.add_edge(e[0], e[1])
        # edges = [(e[0], e[1]) for e in g.edges]
        # print('------+++++++++', g.edges)
        # g_tmp.add_vertices(list(g.nodes))
        # g_tmp.add_edges(edges)
    else:
        g, filtered_adj_dict = get_origin_graph_by_selected_cluster(out_adj_table, data_id)
        if use_igraph is True:
            g = networkx2igraph(g.G)
        else:
            g = g.G

    # print('边 ', lg.edges())
    # print('点 ', lg.nodes())

    #  计算簇中心坐标 ========================================
    # tmp = {}
    # for key in cluster_point_dict:
    #     if int(key) in used_od_cells:
    #         tmp[int(key)] = cluster_point_dict[key]
    # # cluster_point_dict = tmp

    # total_od_points = get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)[
    #     'od_points']
    # cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_ids)

    # # +++++++++++++++ 轨迹获取和特征 ++++++++++++++
    # # node_label_dict = None
    # if os.path.exists(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl'):
    #     with open(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl', 'rb') as f:
    #         obj = pickle.loads(f.read())
    #         trj_idxs, node_names_trjId_dict = obj['trj_idxs'], obj['node_names_trjId_dict']
    # else:
    #     trj_idxs, node_names_trjId_dict = get_trj_ids_by_force_node(force_nodes, cluster_point_dict, total_od_points, region)
    #     with open(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl', 'wb') as f:
    #         picklestring = pickle.dumps({
    #             'trj_idxs': trj_idxs,
    #             'node_names_trjId_dict': node_names_trjId_dict
    #         })
    #         f.write(picklestring)
    #         f.close()
    # print('get_trj_ids_by_force_node')
    '''
    if os.path.isfile(args.best_model):
        print("=> loading best_model '{}'".format(args.best_model))
        best_model = torch.load(args.best_model)
    '''
    # print('trj len', len(trj_idxs))
    # print('node name len', len(node_names_trjId_dict.keys()))
    node_names_trjFeats_dict = {}   # 节点名 -> 包含的轨迹特征数组的 map
    # trjId_node_name_dict = {}   # 轨迹ID -> 所在的节点名的 map
    # node_names_trj_dict = {}    # 节点名 -> gps 轨迹数组的 map
    if use_line_graph:
        if os.path.isfile(args['best_model']):
            print("=> loading best_model '{}'".format(args['best_model']))
            best_model = torch.load(args['best_model'], map_location='cpu')
        with open(f"../data/selected_od_trj_dict_{data_id}.pkl", 'rb') as file:
            selected_od_trj_dict = pickle.loads(file.read())
        for od_pair in selected_od_trj_dict.keys():
            gps_trips = selected_od_trj_dict[od_pair]
            gps_trips = [trip[2:] for trip in gps_trips]
            features = get_trj_feats(gps_trips, best_model)
            node_names_trjFeats_dict[od_pair] = features
        print('node_names_trjFeats_dict', node_names_trjFeats_dict)

    # for node_name in node_names_trjId_dict:
    #     node_trj_idxs = node_names_trjId_dict[node_name]
    #     for trj_id in node_trj_idxs:
    #         trjId_node_name_dict[trj_id] = node_name

    # trj_idxs = list(trjId_node_name_dict.keys())  # 所有轨迹id, trjId 的形式为 {天}_{当天的轨迹id}，这是由于每新的一天，轨迹id都从0开始算
    # gps_trips = get_trips_by_ids(trj_idxs, month, start_day, end_day)
    #
    # print('draw_cluster_in_trj_view======================')
    # draw_cluster_in_trj_view([1 for i in range(len(gps_trips))], gps_trips)
    trj_feats = get_trj_feats    # 特征数组，顺序与 trj_idxs 对应
    # print(f'轨迹id数= {len(trj_idxs)}, 轨迹数 = {len(gps_trips)}, 特征数 = {len(trj_feats)}')

    # for i in range(len(trj_idxs)):
    #     id = trj_idxs[i]
    #     feat = trj_feats[i]
    #     trip = gps_trips[i]
    #     node_name = trjId_node_name_dict[id]
    #     if node_name not in node_names_trjFeats_dict:
    #         node_names_trjFeats_dict[node_name] = []
    #         node_names_trj_dict[node_name] = []
    #     node_names_trjFeats_dict[node_name].append(feat)    # 得到每个节点对应的其包含的特征们
    #     node_names_trj_dict[node_name].append(trip)

    # total_num = 0
    # for name in node_names_trjFeats_dict:
    #     total_num += len(node_names_trjFeats_dict[name])
    #     # print(f"{name} 包含 {len(node_names_trjFeats_dict[name])} 条轨迹")
    # avg_num = total_num // len(node_names_trjFeats_dict.keys())

    # ============== GCC 社区发现代码 ===============
    if use_line_graph:
        adj_mat = get_adj_matrix(g)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
        features, related_node_names = get_feature_list(lg, node_names_trjFeats_dict, 0)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name

    # print(f'原图节点个数：{len(g.nodes())}')
    # print('向量长度', len(features[0]))

    is_none_graph_baseline = False
    is_none_feat_baseline = False

    # print(f'===>> g.nodes = {g.nodes}')
    # related_node_names = list(g.nodes())
    # print(list(g.nodes()))
    # if is_none_feat_baseline is True:
    #     shape = [768]  # features[0].shape
    #     # print(features[0])
    #     features = []
    #     # related_node_names = []
    #     for node in g.nodes():
    #         features.append(np.random.random(shape))
    #         # print(f'---> g.nodes[i] = {g.nodes[node]}')
    #         # related_node_names.append(g.nodes[node])
    #         # features.append(np.zeros(shape))
    #     features = np.array(features)
    #     # print(features[0])

    ######## 仅在做实验时需要这个 for 循环，否则不需要循环，执行一次即可\
    tsne_points = []
    cluster_point_dict = {}
    # for cluster_num in [10, 20, 30, 40, 50]:
    for cluster_num in [4, 4, 5]:
        if is_none_graph_baseline:
            pass
            # labels_dict, trj_labels = get_cluster_by_trj_feature(cluster_num, torch.from_numpy(features))
            # # print('labels_dict==============t', labels_dict)
            # tsne_points = utils.DoTSNE_show(features, 2, trj_labels)
            # print('tsne_points', len(tsne_points))
            # # print('labels_dict', labels_dict)
            # node_name_cluster_dict = {}
            # for i in labels_dict:
            #     label = labels_dict[i]
            #     if label not in cluster_point_dict:
            #         cluster_point_dict[label] = []
            #     # 在线图中度为 0 的散点，视为噪声，从社区中排除
            #     # if get_degree_by_node_name(lg, related_node_names[i]) > 0:
            #     cluster_point_dict[label].append(related_node_names[int(i)])
            #     node_name_cluster_dict[related_node_names[int(i)]] = label
            # print('实际有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            # exp3_log.append(f'实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
        else:
            weight = 'edge_feature' if consider_edge_weight is True else None
            # louvain --------------------------------------------------------------------
            if not use_line_graph:
                # communities = nx.algorithms.community.louvain_partitions(g, weight=weight, resolution=1.6, threshold=1e-06, seed=31)
                # print('=====> communities1=', communities)
                # trj_labels = []
                # for c in communities:
                #     print('c ===', c)
                #     trj_labels.append(c)
                # print('trj==', trj_labels)
                # communities = trj_labels[0]
                # print('=====> communities2=', communities)
                # node_name_cluster_dict = {}
                # cluster_point_dict = {}
                # for (i, cluster) in enumerate(communities):
                #     cluster_point_dict[i] = list(cluster)
                #     for cluster_id in cluster:
                #         node_name_cluster_dict[cluster_id] = i

            # em --------------------------------------------------------------------------
            # communities = algorithms.em(g, cluster_num)
            # communities = algorithms.async_fluid(g, cluster_num)
            # g_tmp = ig.Graph(directed=True)
            # g_tmp.add_vertices(list(g.nodes))
            # g_tmp.add_edges(list(g.edges))
            # g = g_tmp
            # communities = g.community_edge_betweenness(clusters=cluster_num, directed=True, weights=None)
            # print('=====> communities1=', communities)
            # trj_labels = communities
            # communities = list(communities.communities)
            # node_name_cluster_dict = {}
            # cluster_point_dict = {}
            # for (i, cluster) in enumerate(communities):
            #     cluster_point_dict[i] = list(cluster)
            #     for cluster_id in cluster:
            #         node_name_cluster_dict[cluster_id] = i

            # community_edge_betweenness (igraph)  ------------------------------------------------------
            # com = g.community_edge_betweenness(clusters=cluster_num, directed=True, weights=None)
            # # com = g.community_leading_eigenvector(clusters=cluster_num, arpack_options=None, weights=None)
            # # print('com is ==============>', com)
            # # com = ig.GraphBase.community_edge_betweenness(g, 3, True)
            # print('com ===========>', com.as_clustering())
            # print('com ===========>', com)
            # communities = com.as_clustering()
            # trj_labels = communities
            # # communities = list(communities.communities)
            # node_name_cluster_dict = {}
            # cluster_point_dict = {}
            # for (i, cluster) in enumerate(communities):
            #     cluster_point_dict[i] = list(cluster)
            #     for cluster_id in cluster:
            #         node_name_cluster_dict[cluster_id] = i

            # asyn_lpa_communities --------------------------------------------------------
            # communities = nx.algorithms.community.asyn_lpa_communities(g, weight=weight, seed=None)
            # print('=====> communities1=', communities)
            # trj_labels = []
            # for c in communities:
            #     trj_labels.append(c)
            # print('trj==', trj_labels)
            # communities = trj_labels
            # print('=====> communities2=', communities)
            # node_name_cluster_dict = {}
            # cluster_point_dict = {}
            # for (i, cluster) in enumerate(communities):
            #     cluster_point_dict[i] = list(cluster)
            #     for cluster_id in cluster:
            #         node_name_cluster_dict[cluster_id] = i

            # greedy_modularity_communities --------------------------------------------------------
                communities = nx.algorithms.community.greedy_modularity_communities(g, weight=weight, resolution=1.0, cutoff=1.8, best_n=None)
                print('=====> communities1=', communities)
                trj_labels = []
                for c in communities:
                    trj_labels.append(list(c))
                print('trj==', trj_labels)
                communities = trj_labels
                print('=====> communities2=', communities)
                node_name_cluster_dict = {}
                cluster_point_dict = {}
                for (i, cluster) in enumerate(communities):
                    cluster_point_dict[i] = list(cluster)
                    for cluster_id in cluster:
                        node_name_cluster_dict[cluster_id] = i

            # 本文方法 ----------------------------------------------------------------------
            if use_line_graph:
                trj_labels = run(adj_mat, features, cluster_num)  # 得到社区划分结果，索引对应 features 的索引顺序，值是社区 id
                trj_labels = trj_labels.numpy().tolist()
                node_name_cluster_dict = {}
                for i in range(len(trj_labels)):
                    label = trj_labels[i]
                    if label not in cluster_point_dict:
                        cluster_point_dict[label] = []
                    # 在线图中度为 0 的散点，视为噪声，从社区中排除
                    # if get_degree_by_node_name(lg, related_node_names[i]) > 0:
                    cluster_point_dict[label].append(related_node_names[i])
                    node_name_cluster_dict[related_node_names[i]] = label
                print('实际有效社区个数: ', len(cluster_point_dict.keys()))
                exp3_log.append(f'实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
                print('cluster_point_dict', cluster_point_dict)
                # vis_community(cluster_point_dict, selected_od_trj_dict)
        # print(
        #     f'=========> feat len={len(features)}  nodename len={len(related_node_names)}  label len={len(trj_labels)}')
        # print(list(trj_labels))
        # dag_force_nodes, dag_force_edges = get_dag_from_community(cluster_point_dict, force_nodes)

        # to_draw_trips_dict = {}
        # for label in cluster_point_dict:
        #     to_draw_trips_dict[label] = []
        #     for node_name in cluster_point_dict[label]:
        #         to_draw_trips_dict[label].extend(node_names_trj_dict[node_name])
        # print('to_draw_trips_dict', to_draw_trips_dict)
        # data_dict, od_dict = draw_cluster_in_trj_view_new(to_draw_trips_dict, cluster_num, region)
        # with pd.ExcelWriter(f'./cluster_res/excel/{start_day}-{end_day}-{start_hour}-{end_hour}-od_{cluster_num}_cluster_data.xlsx') as writer:
        #     for cluster_id in data_dict:
        #         data_frame = data_dict[cluster_id]
        #         data_frame = pd.DataFrame(data_frame)
        #         data_frame.to_excel(writer, sheet_name=f'社区{cluster_id}', index=False)
        #         od_data_frame = od_dict[cluster_id]
        #         od_data_frame = pd.DataFrame(od_data_frame)
        #         od_data_frame.to_excel(writer, sheet_name=f'社区{cluster_id}_od点', index=False)
        # tsne_points = utils.DoTSNE(features, 2, cluster_point_dict)
        # print(len(g.nodes))

        # print(f'====> 社区个数：{cluster_num}, Q = {Q(lg, node_name_cluster_dict)}')
        print(f'====> 社区个数：{cluster_num}, CON = {avg_CON(g, cluster_point_dict, node_name_cluster_dict, use_igraph)}')
        # print(f'====> 社区个数：{cluster_num}, TPR = {avg_TPR(lg, cluster_point_dict)}')

    if is_none_graph_baseline:
        file_name = f'../result/{exp3_log_name}_none_graph_baseline_Q.txt'
    elif is_none_feat_baseline:
        file_name = f'../result/{exp3_log_name}_none_feat_baseline_Q.txt'
    else:
        file_name = f'../result/{exp3_log_name}_our_Q.txt'
    f = open(file_name, 'w')
    for log in exp3_log:
        f.write(log + '\n')
    f.close()

    # return {
    #     # 'force_nodes': force_nodes,
    #     # 'force_edges': force_edges,
    #     'filtered_adj_dict': filtered_adj_dict,
    #     'cid_center_coord_dict': cid_center_coord_dict,
    #     'community_group': cluster_point_dict,
    #     'tsne_points': tsne_points,
    #     'trj_labels': trj_labels,  # 每个节点（OD对）的社区label，与 tsne_points 顺序对应
    #     'related_node_names': related_node_names,
    #     'tmp_trj_idxs': related_node_names,  # 与 tsne_points 顺序对应,
    #     'node_name_cluster_dict': node_name_cluster_dict
    #     # 'tid_trip_dict': tid_trip_dict,
    # }


if __name__ == '__main__':
    def print_time():  # 无限循环
        print('-------------------->>>>  start')
        while True:  # 获取当前的时间
            current_time = time.ctime(time.time())  # 输出线程的名字和时间
            print('keep live', current_time)  # 休眠10分钟，即600秒 time.sleep(600)
            time.sleep(600)


    # thread = threading.Thread(target=print_time)
    # thread.start()

    od_region = get_region()
    # cell_id_center_coord_dict = get_cell_id_center_coord_dict(od_region)
    # for key in cell_id_center_coord_dict:
    #     print(key, cell_id_center_coord_dict[key])
    # with open("../region.pkl", 'rb') as file:
    #     trj_region = pickle.loads(file.read())
    # makeVocab(trj_region, h5_files)
    # total_od_pairs = get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, od_region)
    # print(total_od_pairs[0:3])
    # od_pairs, od_cell_set, od_pair_set, hot_od_gps_set = get_od_hot_cell(total_od_pairs, od_region, 1000, 0)
    data_id = 20
    out_adj_table = get_grid_split(data_id)
    trj_region = None
    get_line_graph(od_region, trj_region, out_adj_table, data_id)
