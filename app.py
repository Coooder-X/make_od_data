import os
import pickle

from flask import Flask, request
from flask_cors import CORS
import json

import od_pair_process
from spatial_grid_utils import get_region, encode_od
from visualization import increase_filename, draw_trj, get_label_color_dict

app = Flask(__name__)
CORS(app, resources=r'/*')

global_data = {}


@app.route('/getODPointsFilterByDayAndHour', methods=['get', 'post'])
def get_od_points_filter_by_day_and_hour():
    month = request.args.get('month', 5, type=int)
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(month, start_day, end_day, start_hour, end_hour)
    # return json.dumps({'od_points':od_pair_process.get_hour_od_points()})
    return json.dumps({month: od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)})


@app.route('/getTrjsByTime', methods=['get', 'post'])
def get_trjs_by_day_and_hour():
    month = 5
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(f'[info] getTrjsByTime: {month}, {start_day}, {end_day}, {start_hour}, {end_hour}')
    total_trips = od_pair_process.get_trj_num_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)
    return json.dumps(len(total_trips))


@app.route('/getODGraphByTime', methods=['get', 'post'])
def get_od_graph_by_time():
    month = 5
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(f'[info] getTrjsByTime: {month}, {start_day}, {end_day}, {start_hour}, {end_hour}')
    region = get_region()
    od_list, trj_list = od_pair_process.get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, region)
    print(f'[info] len od_list, trj_list', len(od_list), len(trj_list))
    od_set, od_trj_dict, od_num_dict = od_pair_process.get_od_graph(od_list, trj_list, region)
    global_data['od_trj_dict'] = od_trj_dict
    global_data['od_num_dict'] = od_num_dict
    global_data['od_set'] = od_set
    return json.dumps(od_num_dict)


@app.route('/saveSelectedData', methods=['post'])
def save_selected_data():
    data = request.get_json(silent=True)
    selected_pair = data['pairs']
    od_trj_dict = global_data['od_trj_dict']
    # 保存前端的json数据
    json_str = json.dumps(selected_pair)
    # 将 JSON 字符串写入文件
    with open(increase_filename('od_graph', 'json', 1, 'json_data'), 'w') as f:
        f.write(json_str)

    selected_od_trj_dict = {}
    for pair in selected_pair:
        pair_name = encode_od(pair['start'], pair['end'], '_')
        selected_od_trj_dict[pair_name] = od_trj_dict[pair_name]

    # 轨迹数据结构：
    # [[index], [date], [lon1, lat1, time1], ..., [lon, lat, time]]
    data_target_path = increase_filename('selected_od_trj_dict', 'pkl', 1, 'data')
    with open(data_target_path, 'wb') as f:
        picklestring = pickle.dumps(selected_od_trj_dict)
        f.write(picklestring)
    label_color_dict = get_label_color_dict(selected_od_trj_dict)
    draw_trj(selected_od_trj_dict, label_color_dict, type='trj')
    draw_trj(selected_od_trj_dict, label_color_dict, type='od')
    return 'save success'


@app.route('/getHistoryFileList', methods=['get', 'post'])
def get_history_file_list():
    directory = './json_data'
    file_list = os.listdir(directory)
    return file_list


@app.route('/getHistoryFile', methods=['get', 'post'])
def get_history_file():
    filename = request.args.get('filename', type=str)
    directory = './json_data'
    file_path = os.path.join(directory, filename)
    with open(file_path) as file:
        # 读取配置文件
        json_data = json.load(file)
    return json_data


@app.route('/')
def hello_world():  # put application's code here
    print('okkkk')
    return 'Hello World!'


if __name__ == '__main__':
    print('start')
    # od_pair_process.get_od_points_filter_by_day_and_hour(5, 1, 2, 8, 10)
    app.run(port=5000, host='0.0.0.0')
    # region = get_region()
    # cache.set('region', region)
    # # ==_asdfas2
    # _thread.start_new_thread(app.run(port=5000, host='0.0.0.0'))