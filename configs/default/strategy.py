from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.prefix = "baseline"

# setting for loader
strategy_cfg.sample_method = "random"

strategy_cfg.batch_size = 8
strategy_cfg.p_size = 4
strategy_cfg.k_size = 2

# setting for loss
strategy_cfg.classification = True
strategy_cfg.triplet = False
strategy_cfg.Teaching = False
strategy_cfg.RGB_Teaching = False
strategy_cfg.IR_Teaching = False
strategy_cfg.center = False

# training
strategy_cfg.RGB_Training = False
strategy_cfg.IR_Training = False
strategy_cfg.Student_Training = False

# setting for metric learning
strategy_cfg.margin = 0.3
strategy_cfg.weight_KL = 3.0
strategy_cfg.weight_sid = 1.0
strategy_cfg.weight_sep = 1.0
strategy_cfg.update_rate = 1.0

# settings for optimizer
strategy_cfg.optimizer = "sgd"
strategy_cfg.lr = 0.1
strategy_cfg.wd = 5e-3 #5e-3
##5e-4
strategy_cfg.lr_step = [40]

# 是否使用混合精度训练
strategy_cfg.fp16 = False

strategy_cfg.num_epoch = 60

# settings for dataset
strategy_cfg.dataset = "sysu"
strategy_cfg.image_size = (384, 128)

# settings for augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = False
strategy_cfg.padding = 10

# settings for base architecture
strategy_cfg.drop_last_stride = False
strategy_cfg.pattern_attention = False
strategy_cfg.modality_attention = 0
strategy_cfg.mutual_learning = False
strategy_cfg.rerank = False
strategy_cfg.num_parts = 6

# logging
strategy_cfg.eval_interval = -1
strategy_cfg.start_eval = 60
strategy_cfg.log_period = 10

# testing
strategy_cfg.resume = ''
#/home/zhang/E/RKJ/MAPnet/MPA-LL2-cvpr/checkpoints/regdb/RegDB/model_best.pth
#/root/MPANet/MPA-LL2-cvpr/checkpoints/sysu/SYSU/model_best.pth
#/root/MPANet/MPA-LL2-cvpr/checkpoints/llcm/LLCM/model_best.pth
#/home/zhang/E/RKJ/MAPnet/MPA-cvpr/checkpoints/llcm/LLCM/model_best.pth