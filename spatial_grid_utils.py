from matplotlib import collections as mc
import numpy as np
from matplotlib import pyplot as plt

from SpatialRegionTools import SpatialRegion, get_cell_id_center_coord_dict, gps2cell
# from visualization import randomcolor

cellsizeX = 9284.045532159507 / 10
cellsizeY = 8765 / 10
mid_lon, mid_lat = 120.29986, 30.41829
cityname = "hangzhou"
timecellsize = 120
interesttime = 1100.0
nointeresttime = 3300.0
needTime = 0
delta = 1.0
timefuzzysize = 1
timestart = 0
use_grid = 1
has_label = 0

def get_region():
    # min_lon, min_lat, max_lon, max_lat = get_lon_lat_scope(mid_lon, mid_lat, region_size)
    min_lon, min_lat, max_lon, max_lat = 120.1088, 30.2335, 120.1922, 30.3015

    region = SpatialRegion(cityname,
                           min_lon, min_lat,  # 整个hz
                           max_lon, max_lat,  # 整个hz
                           0, 86400,  # 时间范围,一天最大86400(以0点为相对值)
                           cellsizeX, cellsizeY,
                           timecellsize,  # 时间步
                           1,  # minfreq 最小击中次数
                           40_0000,  # maxvocab_size
                           30,  # k
                           4,  # vocab_start 词汇表基准值
                           interesttime,  # 时间阈值
                           nointeresttime,
                           delta,
                           needTime,
                           2, 4000,
                           timefuzzysize, timestart,
                           hulls=None, use_grid=use_grid, has_label=has_label)

    return region


def encode_od(o, d, connect='_'):
    return f'{o}{connect}{d}'


def decode_od(od, connect='_'):
    o, d = od.split(connect)
    return int(o), int(d)


# def draw_cluster_in_trj_view_new(to_draw_trips_dict, cluster_num, od_region):
#     label_color_dict = {}
#     for label in to_draw_trips_dict:
#         label_color_dict[label] = randomcolor()
#
#     total_data_dict = {}
#     total_od_dict = {}
#
#     cell_id_center_coord_dict = get_cell_id_center_coord_dict(od_region)
#     # fig = plt.figure(figsize=(20, 10))
#     # ax = fig.subplots()
#     idx = 0
#     for label in to_draw_trips_dict:
#
#         data_dict = {'line_name': [],
#                      'index': [],
#                      'lon': [],
#                      'lat': []}
#         od_dict = {'lon': [],
#                      'lat': []}
#
#         fig = plt.figure(figsize=(20, 10))
#         ax = fig.subplots()
#         gps_trips = to_draw_trips_dict[label]
#         lines = []
#         for (i, trip) in enumerate(gps_trips):
#             line = []
#             # for j in range(len(trip) - 1):
#             head_cell_id = gps2cell(od_region, trip[0][0], trip[0][1])
#             tail_cell_id = gps2cell(od_region, trip[-1][0], trip[-1][1])
#             # print(f'===> cell id ={head_cell_id, tail_cell_id}')
#             line.append([cell_id_center_coord_dict[head_cell_id], cell_id_center_coord_dict[tail_cell_id]])
#             lines.append(line)
#             # 添加线起点
#             idx += 1
#             data_dict['line_name'].append(f'line{i}')
#             data_dict['index'].append(idx)
#             od_dict['lon'].append(trip[0][0])
#             od_dict['lat'].append(trip[0][1])
#             data_dict['lon'].append(cell_id_center_coord_dict[head_cell_id][0])
#             data_dict['lat'].append(cell_id_center_coord_dict[head_cell_id][1])
#             # 添加线终点
#             idx += 1
#             data_dict['line_name'].append(f'line{i}')
#             data_dict['index'].append(idx)
#             od_dict['lon'].append(trip[-1][0])
#             od_dict['lat'].append(trip[-1][1])
#             data_dict['lon'].append(cell_id_center_coord_dict[tail_cell_id][0])
#             data_dict['lat'].append(cell_id_center_coord_dict[tail_cell_id][1])
#             # data_dict['index'].append(index)
#         for index, line in enumerate(lines):
#             color = label_color_dict[label]
#             lc = mc.LineCollection(line, colors=color, linewidths=2)
#             ax.add_collection(lc)
#         for index, trip in enumerate(gps_trips):
#             trip = np.array(trip)
#             color = label_color_dict[label]
#             ax.scatter(trip[0][0], trip[0][1], s=8, c=color, marker='o')
#             ax.scatter(trip[-1][0], trip[-1][1], s=8, c=color, marker='o')
#
#         ax.set_xlabel('lon')  # 画出坐标轴
#         ax.set_ylabel('lat')
#         plt.savefig(f'./cluster_res/img/trj_cluster_result{cluster_num}_社区{label}.png')
#         plt.close()
#
#         total_data_dict[label] = data_dict
#         total_od_dict[label] = od_dict
#     return total_data_dict, total_od_dict