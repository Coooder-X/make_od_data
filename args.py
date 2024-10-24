args = {}
args['best_model'] = 'D:/PycharmProjects/make_od_data/best_model_4.pt'
args['best_cluster_model'] = 'best_cluster_model_4.pt'
args['cluster_model'] = 'cluster_model_4.pt'
args['pretrained_embedding'] = None
args['num_layers'] = 3
args['bidirectional'] = True
args['hidden_size'] = 256
args['embedding_size'] = 512
args['dropout'] = 0.1
args['max_grad_norm'] = 5.0
args['learning_rate'] = 0.002
args['m2_learning_rate'] = 0.008
args['batch'] = 16
args['generator_batch'] = 32
args['t2vec_batch'] = 128
args['start_iteration'] = 0
args['epochs'] = 100
args['print_freq'] = 40
args['save_freq'] = 40
args['cuda'] = False
# args[']#']   criterion_name='KLDIV'
# args[']#']   knearestvocabs='data\\hangzhou-vocab-dist-cell-120.h5'
args['dist_decay_speed'] = 0.8
args['max_num_line'] = 20000000
args['max_length'] = 200
args['mode'] = 9
args['vocab_size'] = 247415
args['bucketsize'] = [(100000, 100000)]
args['clusterNum'] = 110
args['alpha'] = 1
args['beta'] = 1
args['gamma'] = 1
args['delta'] = 1
args['sourcedata'] = 'pre_data/*.h5'
args['devices'] = [0, 1, 2, 3, 4, 5, 6, 7]
args['expId'] = 4
args['device'] = 3
args['save'] = True
args['hasLabel'] = False
args['kmeans'] = 0