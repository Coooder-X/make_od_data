from graph_process.Graph import Graph
from graph_process.Point import Point


def get_line_graph_by_selected_cluster(selected_cluster_ids_in_brush, selected_cluster_ids, out_adj_dict):
    """
    :param selected_cluster_ids_in_brush: 一个数组，存储地图中选取框框内的簇的id
    :param selected_cluster_ids: 一个数组，存储地图中已选的所有簇的id
    :param out_adj_dict: 当天数据中所有簇的全量的邻接表，out_adj_dict[x] 存储簇 id 为 x 的簇，会到达的簇的 id 数组
    :exp_od_pair_set: 一个set，包含一些OD对，只有在这个集合中的OD对才可以被用于建图
    :return force_nodes: 转换成的线图的节点数组
    :return force_edges: 转换成的线图的边数组
    :return filtered_adj_dict: 根据已选簇，从全量簇的邻接表中过滤出的已选簇的邻接表
    :return line_graph: 转换后的线图，为 networkx 对象
    """
    # selected_cluster_ids, out_adj_dict = cids, adj
    # selected_cluster_ids = list(set(selected_cluster_ids))
    # print(selected_cluster_ids)
    # 过滤出邻接表中有用的部分
    filtered_adj_dict = {}  # 用已选簇id过滤后的邻接表，索引是簇id
    for cid in selected_cluster_ids:
        if cid not in filtered_adj_dict:
            filtered_adj_dict[cid] = []
        # 如果 to_cid 是 cid 的邻接点，则应该加入【过滤邻接表】中
        for to_cid in selected_cluster_ids:
            if to_cid == cid:
                continue
            # 如果起终点都不在地图选取框框内的，就过滤掉
            if cid not in selected_cluster_ids_in_brush and to_cid not in selected_cluster_ids_in_brush:
                continue
            # if cid in out_adj_dict and to_cid in out_adj_dict[cid] and \
            #         (cid, to_cid) in exp_od_pair_set:
            if cid in out_adj_dict and to_cid in out_adj_dict[cid]:
                filtered_adj_dict[cid].append(to_cid)

    cluster_list = []   # 存储所有 Point 类型的 簇，作为 graph 的节点集
    cid_point_dict = {}  # 簇id 到 Point 类型的簇 的映射

    for cid in selected_cluster_ids:
        point = Point(name=cid, nodeId=cid, infoObj={}, feature={})
        cluster_list.append(point)
        cid_point_dict[cid] = point

    adj_point_dict = {}  # 根据 filtered_adj_dict 得出的等价的邻接表，索引是 Point 类型的簇
    for cid in filtered_adj_dict:
        point = cid_point_dict[cid]
        if point not in adj_point_dict:
            adj_point_dict[point] = []
        for to_cid in filtered_adj_dict[cid]:
            adj_point_dict[point].append(cid_point_dict[to_cid])

    g = Graph()
    for cluster in cluster_list:
        g.addVertex(cluster)
    #   边权值可以后续改成簇之间的 od 对数量，暂时默认为 1
    for point in adj_point_dict:
        edge = []
        for to_point in adj_point_dict[point]:
            edge.append([to_point, 1])
        g.addDirectLine(point, edge)
    line_graph = g.getLineGraph()
    # g.drawGraph()
    # g.drawLineGraph()
    # print(line_graph.nodes)
    # print(line_graph.edges)
    # print('点数据', line_graph.nodes.data())
    print('点个数', len(line_graph.nodes))
    print('边个数', len(line_graph.edges))

    force_nodes = []
    for node in line_graph.nodes:
        force_nodes.append({ 'name': f'{node[0]}_{node[1]}' })
    force_edges = []
    for edge in line_graph.edges:
        p1, p2 = edge[0], edge[1]
        force_edges.append({ 'source': f'{p1[0]}_{p1[1]}', 'target': f'{p2[0]}_{p2[1]}' })
    return force_nodes, force_edges, filtered_adj_dict, line_graph