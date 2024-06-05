import os
import pickle
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import collections as mc

from SpatialRegionTools import get_cell_id_center_coord_dict, gps2cell
from global_param import project_path
from spatial_grid_utils import get_region, decode_od


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def get_label_color_dict(to_draw_trips_dict):
    label_color_dict = {}
    for label in to_draw_trips_dict:
        # 方向相反的OD对，轨迹颜色应当相同
        o, d = decode_od(label, connect='_')
        if f'{d}_{o}' in label_color_dict:
            print('[info]', o, d)
            label_color_dict[label] = label_color_dict[f'{d}_{o}']
        else:
            label_color_dict[label] = randomcolor()
    print(len(label_color_dict.keys()))
    print(label_color_dict.keys())
    print(label_color_dict.values())
    return label_color_dict


def draw_trj(to_draw_trips_dict, label_color_dict, type='trj'):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    for label in to_draw_trips_dict:
        gps_trips = to_draw_trips_dict[label]
        lines = []
        for (i, trip) in enumerate(gps_trips):
            line = []
            # for j in range(len(trip) - 1):
            trip = trip[2:] ##############################################
            if type == 'trj':
                for j in range(len(trip) - 1):
                    line.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])
            elif type == 'od':
                line.append([(trip[0][0], trip[0][1]), (trip[-1][0], trip[-1][1])])
            lines.append(line)

        for index, line in enumerate(lines):
            color = label_color_dict[label]
            lc = mc.LineCollection(line, colors=color, linewidths=2)
            ax.add_collection(lc)
        for index, trip in enumerate(gps_trips):
            trip = np.array(trip[2:])##############################################
            color = label_color_dict[label]
            ax.scatter(trip[0][0], trip[0][1], s=8, c=color, marker='o')
            ax.scatter(trip[-1][0], trip[-1][1], s=8, c=color, marker='o')

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')

    filename = 'trj_vis' if type == 'trj' else 'od_vis'
    # begin = 7 if type == 'od' else 1
    plt.savefig(increase_filename(filename, 'png', 1, 'data'))
    plt.close()


def draw_trj2(to_draw_trips_dict, label_color_dict, type='trj', save_excel=False):
    total_data_dict = {}
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    data_idx = 0
    line_idx = 0
    data_dict = {'line_name': [],
                 'index': [],
                 'lon': [],
                 'lat': []}
    for (idx, label) in enumerate(to_draw_trips_dict.keys()):
        gps_trips = to_draw_trips_dict[label]
        lines = []
        for (i, trip) in enumerate(gps_trips):
            # if random.random() < 0.7:
            #     continue
            line = []
            line_idx += 1
            if line_idx in set([86, 311, 502, 464, 230, 408]):
                # line_idx -= 1
                continue
            trip = trip[2:] ##############################################
            if type == 'trj':
                for j in range(len(trip) - 1):
                    line.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])
                if save_excel:
                    for j in range(len(trip)):
                        data_idx += 1
                        data_dict['line_name'].append(f'line{line_idx}')
                        data_dict['index'].append(data_idx)
                        data_dict['lon'].append(trip[j][0])
                        data_dict['lat'].append(trip[j][1])
            elif type == 'od':
                line.append([(trip[0][0], trip[0][1]), (trip[-1][0], trip[-1][1])])
                # plt.arrow(trip[0][0], trip[0][1], trip[-1][0], trip[-1][1],
                #           head_width=0.1, head_length=0.1,
                #           fc=label_color_dict[label], ec=label_color_dict[label])
            lines.append(line)
        # total_data_dict['sheet1'].extend(data_dict)
        print('data_dict', data_dict)
        print('label', label)
        if save_excel:
            continue

        for index, line in enumerate(lines):
            color = label_color_dict[label]
            lc = mc.LineCollection(line, colors=color, linewidths=2)
            ax.add_collection(lc)
        for index, trip in enumerate(gps_trips):
            trip = np.array(trip[2:])##############################################
            color = label_color_dict[label]
            ax.scatter(trip[0][0], trip[0][1], s=8, c=color, marker='o')
            ax.scatter(trip[-1][0], trip[-1][1], s=38, c=color, marker='^')
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')

    if save_excel:
        total_data_dict['sheet1'] = data_dict
        with pd.ExcelWriter(f'./result/trj.xlsx') as writer:
            for key in total_data_dict:
                data_frame = total_data_dict[key]
                data_frame = pd.DataFrame(data_frame)
                data_frame.to_excel(writer, sheet_name=f'{key}', index=False)
        print(data_frame)
        return

    filename = 'trj_vis_our' if type == 'trj' else 'od_vis_our'
    begin = 1
    plt.savefig(increase_filename(filename, 'png', begin, 'data/our'))
    plt.close()


def vis_community(cluster_point_dict, selected_od_trj_dict):
    cluster_label = cluster_point_dict.keys()
    color_table = ['#0780cf', '#765005', '#fa6d1d', '#0e2c82', '#b6b51f', '#da1f18', '#701866']
    label_color_dict = {}
    to_draw_trips_dict = {}
    for (idx, label) in enumerate(cluster_label):
        label_color_dict[label] = color_table[idx]
        od_pairs = cluster_point_dict[label]    #  ['35_26', '35_53', '53_35', '61_53', '65_53', '74_53', '53_35']
        to_draw_trips_dict[label] = []
        for pair in od_pairs:
            to_draw_trips_dict[label].extend(selected_od_trj_dict[pair])
    draw_trj2(to_draw_trips_dict, label_color_dict, 'od')
    draw_trj2(to_draw_trips_dict, label_color_dict, 'trj')


def increase_filename(name, filetype, begin=1, path='data'):
    i = begin
    while True:
        filename = f"{project_path}/{path}/{name}_{i}.{filetype}"
        if not os.path.exists(filename):
            return filename
        i += 1


if __name__ == '__main__':
    od_region = get_region()
    with open("./data/selected_od_trj_dict_8.pkl", 'rb') as file:
        selected_od_trj_dict = pickle.loads(file.read())
    label_color_dict = get_label_color_dict(selected_od_trj_dict)
    draw_trj(selected_od_trj_dict, label_color_dict, type='trj')
    draw_trj(selected_od_trj_dict, label_color_dict, type='od')
