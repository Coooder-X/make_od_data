import os
import pickle
from datetime import datetime

from SpatialRegionTools import inregionS, gps2cell
from global_param import use_database, tmp_file_path, project_father_path
from spatial_grid_utils import encode_od


def get_od_points_filter_by_day_and_hour(month, start_day, end_day, start_hour=0, end_hour=24):
    if not use_database:
        od_points = get_total_od_points_by_day(month, start_day, end_day)
        res = []
        index_list = []
        for i in range(0, len(od_points), 2):
            if start_hour * 3600 <= od_points[i][2] <= end_hour * 3600:  # and start_hour * 3600 <= od_points[i + 1][2] <= end_hour * 3600: todo: 这样部分跨时间的OD对会被拆分，有的地方可能会取到
                res.append(od_points[i])
                res.append(od_points[i + 1])
                index_list.append(i)
                index_list.append(i + 1)
        print(f'{start_day}-{end_day} {start_hour}-{end_hour} OD点总数：', len(od_points))
        return {'od_points': res, 'index_lst': index_list}
    # else:
    #     return query_od_points_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)


def get_total_od_points_by_day(month, start_day, end_day):
    res = []
    for i in range(start_day, end_day + 1):
        start_time = datetime.now()
        data_target_path = tmp_file_path + "2020" + str(month).zfill(2) + str(i).zfill(2) + ".pkl"
        data_source_path = project_father_path + str(month) + "月/" + str(month).zfill(2) + "月" + str(i).zfill(
            2) + "日/2020" + str(month).zfill(2) + str(i).zfill(
            2) + "_hz.h5"
        # if not os.path.exists(data_target_path):
        #     filter_step = 1
        #     get_odpoints_and_save_as_pkl_file(data_source_path, data_target_path, filter_step, i)
        with open(data_target_path, 'rb') as file:
            od_points = pickle.loads(file.read())
        print('读取文件结束，用时: ', (datetime.now() - start_time))
        # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
        for (idx, od) in enumerate(od_points):
            res.append(od.tolist())
    print(f'---> res.len = {len(res)}, res[0] = {res[0]}')
    return res


#  获取所有轨迹
def get_trj_num_filter_by_day_and_hour(month, start_day, end_day, start_hour=0, end_hour=24):
    if not use_database:
        trips = get_trj_num_filter_by_day(month, start_day, end_day)
        print(f'[WARN] trips from pkl {trips[0]}')
        part_od_coord_trips, index_list = trips_filter_by_hour(trips, start_hour, end_hour)
    # else:
    #     # trips = query_trips_by_day('trajectory_db', start_day, end_day + 1)
    #     part_od_coord_trips, index_list = query_trj_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)
    #     print(f'[WARN] trips from mysql {part_od_coord_trips[0]}')
    return {'trips': part_od_coord_trips, 'index_lst': index_list}


def trips_filter_by_hour(trips, start_hour, end_hour):
    print(len(trips))
    index_list = []
    res = []
    for i in range(len(trips)):
        if start_hour * 3600 <= trips[i][2][2] <= end_hour * 3600:
            index_list.append(i)
            res.append(trips[i])
    return res, index_list


def get_trj_num_filter_by_day(month, start_day, end_day):
    res = []
    start_time = datetime.now()
    for i in range(start_day, end_day + 1):
        data_target_path = tmp_file_path + "2020" + str(month).zfill(2) + str(i).zfill(2) + "_trj.pkl"
        data_source_path = project_father_path + str(month) + "月/" + str(month).zfill(2) + "月" + str(i).zfill(
            2) + "日/2020" + str(month).zfill(2) + str(i).zfill(
            2) + "_hz.h5"
        with open(data_target_path, 'rb') as file:
            od_points = pickle.loads(file.read())
        print('读取文件结束，用时: ', (datetime.now() - start_time))
        # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
        for (idx, od) in enumerate(od_points):
            res.append(od)
    return res


def get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, region):
    # =============== 旧的实现 ==================
    if not use_database:
        od_list = []
        trj_list = []
        start_time = datetime.now()
        for i in range(start_day, end_day + 1):
            data_target_path = tmp_file_path + "2020" + str(month).zfill(2) + str(i).zfill(2) + "_trj.pkl"
            data_source_path = project_father_path + str(month) + "月/" + str(month).zfill(2) + "月" + str(i).zfill(
                2) + "日/2020" + str(month).zfill(2) + str(i).zfill(
                2) + "_hz.h5"
            if not os.path.exists(data_target_path):
                pass
            with open(data_target_path, 'rb') as file:
                trjs = pickle.loads(file.read())
                # print(i, trjs[0])
            print('读取文件结束，用时: ', (datetime.now() - start_time))
            # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
            for (idx, trj) in enumerate(trjs):
                t = trj[2:]
                o, d = t[0], t[-1]
                if inregionS(region, o[0], o[1]) and inregionS(region, d[0], d[1]) and \
                        start_hour * 3600 <= o[2] <= end_hour * 3600 and\
                        start_hour * 3600 <= d[2] <= end_hour * 3600:
                    od_list.append([o, d])   # trj[0] 是轨迹的id，trj[1]是日期，trj[2:]是轨迹点序列
                    trj_list.append(trj)
                    #  保持OD与轨迹的一对一映射关系
    # 这里的OD是轨迹起终点的坐标，不是网格id
    return od_list, trj_list


def get_od_graph(od_list, trj_list, region):
    od_set = set()
    od_trj_dict = {}
    od_num_dict = {}
    for (od, trj) in zip(od_list, trj_list):
        o, d = od[0], od[1]
        o, d = gps2cell(region, o[0], o[1]), gps2cell(region, d[0], d[1])
        pair = encode_od(o, d, connect='_')
        if pair not in od_set:
            od_set.add(pair)
            od_trj_dict[pair] = []
            od_num_dict[pair] = 0
        od_trj_dict[pair].append(trj)
        od_num_dict[pair] += 1
    return od_set, od_trj_dict, od_num_dict


def get_trj_ids_by_force_node(force_nodes, part_cluster_point_dict, total_od_points):
    trj_ids = []
    node_names_trjId_dict = {}
    i = 0
    num = len(force_nodes)
    for node in force_nodes:
        i += 1
        print(f'{i}/{num}, {node["name"]}')
        src_cid, tgt_cid = list(map(int, node['name'].split('_')))
        src_points, tgt_points = part_cluster_point_dict[src_cid], part_cluster_point_dict[tgt_cid]
        for src_pid in src_points:
            o = total_od_points[src_pid]
            # print('o在区域内', inregionS(region, o[0], o[1]))
            for tgt_pid in tgt_points:
                d = total_od_points[tgt_pid]
                # print('d在区域内', inregionS(region, d[0], d[1]))
                if o[5] == d[5] and o[3] == d[3] and o[4] == d[4] - 1:
                    trj_ids.append(int(d[3]))
                    if node['name'] not in node_names_trjId_dict:
                        node_names_trjId_dict[node['name']] = []
                    #   trjId 的形式为 {天}_{当天的轨迹id}，这是由于每新的一天，轨迹id都从0开始算
                    node_names_trjId_dict[node['name']].append(encode_trjId(d[5], d[3]))
                    # 粗略的初版实现：只在一个OD对中取一条轨迹。后续改成，一个OD对中多条轨迹都加入，然后特征取平均
    return trj_ids, node_names_trjId_dict


def encode_trjId(day, day_index):
    return f'{int(day)}_{int(day_index)}'


def decode_trjId(trjId):
    day, day_index = trjId.split('_')
    return int(day), int(day_index)