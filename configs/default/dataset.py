from yacs.config import CfgNode

dataset_cfg = CfgNode()
# 实例化 CfgNode 类，创建一个配置节点对象 dataset_cfg。
# 这个对象用于存储与数据集相关的配置参数，比如数据集路径、批量大小、预处理方法等。

# config for dataset
dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = "/home/qzs/Try_person_ReID/dataset/SYSU-MM01"

dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 412
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = "D:/桌面/Try_person_ReID/dataset/RegDB"
# "/home/qzs/Try_person_ReID/dataset/RegDB"
dataset_cfg.llcm = CfgNode()
dataset_cfg.llcm.num_id = 713
dataset_cfg.llcm.num_cam = 2
dataset_cfg.llcm.data_root = "D:/桌面/Try_person_ReID/dataset/RegDB"

