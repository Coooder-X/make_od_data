import os
import pickle

import torch
import od_pair_process
from args import args
from t2vec import run_model2
from visualization import draw_trj2, vis_community


def test_gpu_model():
    best_model = './best_model_4.pt'
    if os.path.isfile(best_model):
        print("=> loading best_model '{}'".format(best_model))
        # best_model = torch.load(best_model)
        best_model = torch.load(best_model, map_location='cpu')
        # print('best_model', best_model)
        with open("./region.pkl", 'rb') as file:
            trj_region = pickle.loads(file.read())
        with open("./data/selected_od_trj_dict_9.pkl", 'rb') as file:
            selected_od_trj_dict = pickle.loads(file.read())
        # key = list(selected_od_trj_dict.keys())[3]
        # gps_trips = selected_od_trj_dict[key]
        gps_trips = []
        for trips in list(selected_od_trj_dict.values()):
            print('trips', len(trips))
            for trip in trips:
                gps_trips.append(trip)
        print('gps_trips', len(gps_trips))
        # gps_trips = [trip for trip in [trips for trips in list(selected_od_trj_dict.values())]]
        gps_trips = [trip[2:] for trip in gps_trips]
        # print('key', key)
        print(len(gps_trips))
        # print(gps_trips[0])
        trj_feats = run_model2(args, gps_trips, best_model, trj_region)
        print('trj_feats', trj_feats)


def extend_data(data_id, tar_od_pairs):
    origin_file_name = f'./data/selected_od_trj_dict_{data_id}_origin.pkl'
    new_file_name = f'./data/selected_od_trj_dict_{data_id}.pkl'
    with open(origin_file_name, 'rb') as file:
        selected_od_trj_dict = pickle.loads(file.read())
    print(selected_od_trj_dict)
    # for pair in tar_od_pairs:
    #     tmp = selected_od_trj_dict[pair]
    #     while len(selected_od_trj_dict[pair]) < 23:
    #         selected_od_trj_dict[pair].extend(tmp[:5])
    with open(new_file_name, 'wb') as f:
        picklestring = pickle.dumps(selected_od_trj_dict)
        f.write(picklestring)
    return


if __name__ == '__main__':
    # print('start')
    # # od_pair_process.get_od_points_filter_by_day_and_hour(5, 1, 2, 8, 10)
    # total_trips = od_pair_process.get_trj_num_filter_by_day_and_hour(5, 1, 2, 8, 10)
    # tmp = total_trips['trips']
    # print(f'---> res.len = {len(tmp)}, res[0] = {tmp[0]}')

    # test_gpu_model()
    # extend_data(10, ['42_33', '33_42', '54_47', '47_54', '54_66', '66_54'])
    data_id = 20
    with open(f"./data/selected_od_trj_dict_{data_id}.pkl", 'rb') as file:
        selected_od_trj_dict = pickle.loads(file.read())
    to_draw_trips_dict = selected_od_trj_dict
    print(list(to_draw_trips_dict.keys())[20])
    # draw_trj2(to_draw_trips_dict, {}, 'trj', True)
    # cnt = 0
    # for key in selected_od_trj_dict:
    #     cnt += len(selected_od_trj_dict[key])
    # print(cnt)
