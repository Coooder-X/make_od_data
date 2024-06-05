import os

import torch

from SpatialRegionTools import trip2seq
from model.data_utils import MyDataOrderScaner
from model.models import EncoderDecoder


def run_model2(args, gps_trips, best_model, trj_region):
    tmp_gps_trips = []
    for trip in gps_trips:
        tmp_trip = []
        for p in trip:
            tmp_trip.append([p[0], p[1]])
        tmp_gps_trips.append(tmp_trip)
    gps_trips = tmp_gps_trips

    vecs = []
    # torch.cuda.set_device(args.device)  # 指定第几块显卡
    # 创建预训练时候用到的模型评估其性能
    m0 = EncoderDecoder(args['vocab_size'], args['embedding_size'],
                        args['hidden_size'], args['num_layers'],
                        args['dropout'], args['bidirectional'])
    # 取出最好的model
    if os.path.isfile(args['best_model']):
        # print("=> loading best_model '{}'".format(args.best_model))
        # best_model = torch.load(args.best_model)
        # # 存时，"m0": m0.state_dict()
        m0.load_state_dict(best_model["m0"])
        if args['cuda'] and torch.cuda.is_available():
            m0.cuda()  # 注意：如果训练的时候用了cuda,这里也必须m0.cuda
        # 如果模型含dropout、batch normalization等层，需要该步骤
        m0.eval()

        # with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/region.pkl", 'rb') as file:
        #     region = pickle.loads(file.read())
        cell_trips = []
        for gps_trip in gps_trips:
            # cell_trj = tripandtime2seq(region, gps_trip)
            cell_trj = trip2seq(trj_region, gps_trip)
            # cell_trj = " ".join(cell_trj)
            cell_trips.append(cell_trj)
        # print('cell_trips', len(cell_trips), cell_trips)
        # torch.cuda.set_device(args.device)  # 指定第几块显卡
        print('len(cell_trips)', len(cell_trips))
        print('cell_trips[0]', cell_trips[0])
        # 初始化需要评估的数据集
        scaner = MyDataOrderScaner(cell_trips, len(cell_trips))
        scaner.load()

        i = 0
        while True:
            if i % 10 == 0:
                print("{}: Encoding {} trjs...".format(i, args['t2vec_batch']))
            i = i + 1
            # 获取到的排序后的轨迹数据、每个轨迹的长度及其排序前的索引
            src, lengths, invp, label = scaner.getbatch_scaner()
            # 没有数据则处理完毕
            if src is None: break
            if args['cuda'] and torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            # 不需要进入m0的前向传播，因为取出了解码器，只进行编码
            h, _ = m0.encoder(src, lengths)
            ## (num_layers, batch, hidden_size * num_directions)
            # 对最后一个hidden的结果进行处理
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions)
            # 转置，并强制拷贝一份tensor
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            # h = h.view(h.size(0), -1)
            # 回到轨迹的原始顺序
            h2 = h[invp]
            size = h2.size()
            h2 = h2.view(size[0], size[1] * size[2])
            # 把3层特征拼接成一个特征
            vecs.append(h2.cpu().data)
        # todo-----------------
        ## (num_seqs, num_layers, hidden_size * num_directions)
        # 把一批批的batch合并, vecs.shape 1899, 3, 256
        # 1899个轨迹
        vecs = torch.cat(vecs)
        # size = vecs.size()
        # 采取不同层的特征
        # feature = vecs[:, 2, :]  # 只提取低3层特征
        # feature1 = vecs[:, 0, :]  # 只提取第1层特征
        # feature2 = vecs[:, 1, :]  # 只提取第2层特征
        # feature = torch.cat((feature2, feature3), 1)  # 合并前两层特征

        # 把3层特征拼接成一个特征
        # feature = vecs.view(size[0], size[1] * size[2])
    # print('vecs', vecs)
    return vecs
