# README

**P.S.** 项目的`dev`分支是干净的可用分支，`master`分支未经整理。

该项目可以分为 2 部分，一部分是作为数据构造工具的后端，一部分是根据构造的数据运行实验，需要手动分开执行。

- `requirement.txt` 在项目根目录下，可安装依赖

### 1、作为数据构造工具的后端
- 运行 app.py，启动后端
- 另一个项目中运行前端
- 在页面中点击保存数据后，后端会将该数据保存为3类：OD对信息、OD对和轨迹映射信息、图片
    - **OD 对信息**：`/json_data/od_graph_{index}.json`，仅包含页面中选择的 OD 流和流量信息
    - **OD对和轨迹映射信息**：`/data/selected_od_trj_dict_{index}.pkl`，文件中存储一个 python 字典对象，其中 key 为形如 `"70_72"` 的字符串，由 "_" 分割两个数字，数字表示地区网格划分的 id，即该 key 表示 "由网格70前往72的OD流"。
    字典的 value 是一个存储轨迹的数组，每个数组元素的结构如下:
    ```[[index], [date], [lon1, lat1, time1], ..., [lon, lat, time]]```
    每个数组元素也是一个数组，表示一条轨迹数据，其中 `index` 是轨迹的 id，`date` 是一个整数，表示该轨迹在2020年5月份中的日期，`[lon1, lat1, time1]` 则表示轨迹的一个采样点。
    - **图片**：分别绘制该数据集的所有轨迹和所有OD对，图片分别命名为 `trj_vis_{index}.png`、 `od_vis_{index}.png`，保存在项目根目录的 `/data` 下。

### 2、运行社区发现实验
运行实验的前置条件为：已经通过构造数据的后端保存了可用的数据，即`/data/selected_od_trj_dict_{index}.pkl`文件。
- 配置文件：`global_param.py` 中配置必要的文件路径，其中`tmp_file_path`对应的是2020年5月份轨迹的`pkl`格式数据（黑色硬盘中or超姐台式机里有），为一个名为`tmp`的文件夹，内部为一系列`pkl`文件。
- `graph_exp.py` 为运行实验的主程序。
#### 运行实验：
- 选择要跑的数据集，例如需要跑`/data/selected_od_trj_dict_20.pkl`这个数据，就在 `graph_exp.py`的 main 函数中配置 data_id = 20，如：
<img src=".\img\20240605210842.png" width="700"/>
- 配置执行实验的选项：在`graph_exp.py`中有如下配置 flag：
  ```python
  consider_edge_weight = True  # 是否考虑边权（在使用传统方法时有效）
  use_line_graph = False  # 是否使用线图方法
  use_igraph = False  # 无用，并始终为 False
  tradition_method = 'CNM'  # 'Louvain' 'CNM'
  draw_cluster = False  # 若执行线图方法，则可以选择是否绘制社区划分结果的图片
  ```
  逻辑如下：
  - **运行我们的线图方法**：将 `use_line_graph` 置 true，其它不管
  - **运行传统方法**：首先将 `use_line_graph` 置 false，并使用`tradition_method`控制使用哪种传统方法。对于是否考虑边的权值，将`consider_edge_weight` 置为对应值。（注：某个传统方法依赖 igraph 库，使用不同的数据结构，因此需要将 `use_igraph`置 true，但我们的论文中没使用该传统方法，因此不管并置 false 即可。）

  本实验会将社区发现算法在数据集上执行3次，对应代码为`graph_exp.py`中的：
  ```python
  for cluster_num in [4, 4, 5]:
      code...
  ```
  其中 for 循环执行3次，即社区发现算法执行3次，cluster_num 仅在 `use_line_graph == True` 时才起作用，因为我们的线图方法需要指定社区个数，社区个数即对应`cluster_num`，做实验时可指定数组中元素的值。
  - 下图中的代码，设置了`tradition_method == 'Louvain'`，运行的是 `networkx.algorithms.community.louvain_partitions` 这个传统方法，
  ```python
    if not use_line_graph:
        # louvain --------------------------------------------------------------------
        if tradition_method == 'Louvain':
            communities = nx.algorithms.community.louvain_partitions(g, weight=weight, resolution=1.6, threshold=1e-06, seed=31)
            trj_labels = []
            for c in communities:
                trj_labels.append(c)
            print('trj==', trj_labels)
            communities = trj_labels[0]
            print('Louvain 社区发现结果为：', communities)
            node_name_cluster_dict = {}
            cluster_point_dict = {}
            for (i, cluster) in enumerate(communities):
                cluster_point_dict[i] = list(cluster)
                for cluster_id in cluster:
                    node_name_cluster_dict[cluster_id] = i
  ```
  而下图中的代码运行的 `networkx.algorithms.community.greedy_modularity_communities` 是对应论文中的 CNM 的传统方法。
  ```python
  # greedy_modularity_communities --------------------------------------------------------
    if tradition_method == 'CNM':
        communities = nx.algorithms.community.greedy_modularity_communities(g, weight=weight, resolution=1.0, cutoff=1.8, best_n=None)
        trj_labels = []
        for c in communities:
            trj_labels.append(list(c))
        print('trj==', trj_labels)
        communities = trj_labels
        print('CNM 社区发现结果为：', communities)
        node_name_cluster_dict = {}
        cluster_point_dict = {}
        for (i, cluster) in enumerate(communities):
            cluster_point_dict[i] = list(cluster)
            for cluster_id in cluster:
                node_name_cluster_dict[cluster_id] = i
  ```
#### 查看实验结果
- 若是运行传统的方法，则包含控制台输出和 log 文件。
  - 控制台输出中，除去无用的 log 外，带有字样 **“社区发现结果为:xxxx”** 一行的输出即为社区发现结果，如：
  `CNM 社区发现结果为:  [[65, 84, 77, 58, 45, 47], [33, 18, 6, 26, 15], [72, 42, 70]]`
  其中 xxxx 部分是一个数组，数组长度是社区个数，数组元素是一个含有多个整数的数组，表示当前社区包含的网格 id。
  - log 文件：`result/exp_log.txt` 中记录了输出的 CON 指标。
- 若运行我们的线图方法，则输出两张图片，位置在 `/data/our`目录下，命名分别是 `trj_vis_our_{index}.png` 和 `od_vis_our_{index}.png`，前者是 OD 对视角下的社区划分，后者是轨迹视角下的社区划分。在图片中，每条轨迹/OD对的终点由小三角形表示，起点由小圆点表示。
- CON 指标输出：对于传统的和我们的方法，都会在控制台输出形如：`result ====> 社区个数：{cluster_num}, CON = {number}`的一行文字，其中 `CON =`后面的数即是 CON 指标，同时也可在`result/exp_log.txt`文件中找到指标的输出。
