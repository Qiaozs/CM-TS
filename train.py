import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from data import get_test_loader
from data import get_train_loader
import torch
import yaml
import logging
import os
import pprint
from models.CM_TS import Net
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from engine import get_trainer


def get_parameter_number(net):
    # 用来统计网络各个分支的参数量
    param_info = {}
    # 遍历模型中的每个子模块（即每个分支）
    for name, module in net.named_children():  # named_children 返回子模块的名称和模块本身
        total_num = sum(p.numel() for p in module.parameters())  # 计算总参数量
        trainable_num = sum(p.numel() for p in module.parameters() if p.requires_grad)  # 可训练参数量
        param_info[name] = {'总参数量': total_num, '可训练参数量': trainable_num}
    return param_info

def get_parameter_number_total(net):  # 用来统计网络参数量
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'总参数量': total_num, '可训练参数量': trainable_num}

def train(cfg):
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
    # 指定日志路径为"logs/cfg.dataset/cfg.prefix"
    # 创建日志文件
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 配置日志记录的基本设置。
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + "log.txt",
                        filemode="w",
                        )
    # %(asctime)s 会插入日志记录的时间戳，%(message)s 是日志内容。

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 将级别设置为 INFO，意味着记录 INFO 级别及以上的日志（INFO，WARNING，ERROR，CRITICAL）
    stream_handler = logging.StreamHandler()  # 创建一个流处理器，将日志输出到控制台
    stream_handler.setLevel(logging.INFO)  # 将流处理器的级别设置为 INFO，控制台将显示 INFO 级别及以上的日志
    logger.addHandler(stream_handler)  # 将处理器添加到日志记录器，使其开始输出日志

    logger.info(pprint.pformat(cfg))
    # 使用 pprint 模块以易读的格式打印 cfg 的内容
    # training data loader
    # get_train_loader()来自data，其指定训练数据集，数据集路径，batch大小等等。
    train_loader = get_train_loader(dataset=cfg.dataset,
                                    root=cfg.data_root,
                                    batch_size=cfg.batch_size,
                                    sample_method= "",
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    image_size=cfg.image_size,
                                    num_workers=0)
    # for batch_idx, (total_batch, rgb_batch, ir_batch) in enumerate(train_loader):
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"  Total batch size: {len(total_batch[0])}")
    #     print(f"  RGB batch size: {len(rgb_batch[0])}")
    #     print(f"  IR batch size: {len(ir_batch[0])}")
    #
    #     if rgb_batch:
    #         print(f"  RGB images in batch: {rgb_batch[0].shape}")  # Expected shape: [batch_size, 3, height, width]
    #     if ir_batch:
    #         print(f"  IR images in batch: {ir_batch[0].shape}")  # Expected shape: [batch_size, 3, height, width]
    #
    #     # Optional: Check the labels and cam_ids to ensure correct mapping
    #     print(f"  RGB labels: {rgb_batch[1]}")
    #     print(f"  IR labels: {ir_batch[1]}")
    #     print(f"  RGB cam_ids: {rgb_batch[2]}")
    #     print(f"  IR cam_ids: {ir_batch[2]}")
    #
    #     # Check for each identity's number of images (expected: k_size // 2 for each)
    #     unique_rgb_labels = set(rgb_batch[1].tolist())
    #     unique_ir_labels = set(ir_batch[1].tolist())
    #     print(f"  Unique RGB labels: {unique_rgb_labels}")
    #     print(f"  Unique IR labels: {unique_ir_labels}")
    #
    #     # Validate batch size per identity
    #     assert len(rgb_batch[0]) == len(ir_batch[0]), f"Mismatch in RGB and IR image count for batch {batch_idx + 1}"
    #
    #
    #     if batch_idx == 1:  # 只检查第一个批次
    #         break  # 退出检查，避免检查所有批次

    # evaluation data loader
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=0)

    # model
    model = Net(num_classes=cfg.num_id,
                p_size=cfg.p_size,
                k_size=cfg.k_size,

                triplet=cfg.triplet,
                Teaching=cfg.Teaching,
                RGB_Teaching=cfg.RGB_Teaching,
                IR_Teaching=cfg.IR_Teaching,

                RGB_Training=cfg.RGB_Training,
                IR_Training=cfg.IR_Training,
                Student_Training=cfg.Student_Training,

                pattern_attention=cfg.pattern_attention,
                modality_attention=cfg.modality_attention,
                mutual_learning=cfg.mutual_learning,
                drop_last_stride=cfg.drop_last_stride,

                margin=cfg.margin,
                num_parts=cfg.num_parts,
                weight_KL=cfg.weight_KL,
                weight_sid=cfg.weight_sid,
                weight_sep=cfg.weight_sep,
                update_rate=cfg.update_rate,
                classification=cfg.classification,

                rerank=cfg.rerank
                )

    # 统计参数量
    print(get_parameter_number_total(model))
    param_info = get_parameter_number(model)
    for branch_name, info in param_info.items():
        print(f"分支 {branch_name}:")
        print(f"  总参数量: {info['总参数量']}")
        print(f"  可训练参数量: {info['可训练参数量']}")

    model.to(device)

    # optimizer
    assert cfg.optimizer in ['adam', 'sgd']
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)

    # 初始化一个 GradScaler 实例，用于支持混合精度训练，替换原文使用的Apex.amp
    scaler = GradScaler(enabled=cfg.fp16)

    # If center loss is used, ensure the centers are in float precision
    if cfg.center:
        model.center_loss.centers = model.center_loss.centers.float()

    # 根据cfg.lr_step调整学习率
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=cfg.lr_step,
                                                  gamma=0.1)

    # 从保存的测试点加载模型权重
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint)
        # cfg.resume 为真即指定了检查点路径，模型的权重和偏置将被恢复到保存时的状态，允许从上次训练中断的地方继续训练。

    # engine
    checkpoint_dir = os.path.join("checkpoints", cfg.dataset, cfg.prefix)

    # get_trainer来自engine，
    engine = get_trainer(dataset=cfg.dataset,
                         model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         start_eval=cfg.start_eval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader,
                         rerank=cfg.rerank)


    # 训练
    engine.run(train_loader, max_epochs=cfg.num_epoch)

if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    # 添加一个名为 --cfg 的命令行参数，类型为字符串（str），默认值为 "configs/softmax.yml"。
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    # 添加一个名为 --gpu 的命令行参数，类型为字符串（str），默认值为 0 ，这个参数用于指定 CUDA 可见设备的 ID。
    '''
    使用命令行指定数据集配置文件，以及GPU_ID：
    python train.py --cfg ./configs/SYSU.yml --gpu 0
    '''
    args = parser.parse_args()
    # 解析传入的命令行参数，并将结果存储在 args 变量中，可以通过 args.cfg 和 args.gpu 来访问这些参数。
    # arg中记录了训练的数据集路径，使用的gpuID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # 将命令行参数 args.gpu 的值赋给环境变量'CUDA_VISIBLE_DEVICES'（ CUDA 可见设备）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("当前设备:", device, "训练数据集：", args.cfg)

    # Set random seed
    seed = 1
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 初始化随机数生成器的值，为随机数生成过程提供一个起始点，从而使得生成的随机数序列是可预测和可复现的。

    # Enable CUDA backend optimizations
    torch.backends.cudnn.benchmark = True
    # 启用 CuDNN 自动优化。根据输入的大小和模型架构来选择最优的算法，在输入大小固定的情况下非常有效，可以加速训练过程。

    # Load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)
    # 使用 yaml 模块读取一个 args.cfg（路径下的） 数据集配置文件（如SYSU.yml），并将其内容加载到 customized_cfg 变量中 “r”为只读模式。
    cfg = strategy_cfg
    # strategy_cfg是configs/default下配置好的内容
    cfg.merge_from_file(args.cfg)
    # 将数据集配置文件内的内容合并到cfg中，同样的部分会被替换为数据集配置文件内的内容，所以优先修改数据集配置文件
    dataset_cfg = dataset_cfg.get(cfg.dataset)
    # dataset_cfg为configs下的自定义默认配置文件，
    # cfg.dataset来自数据集配置文件，为一字符串，如"sysu"。
    # dataset_cfg.get(key)，传入一个键值cfg.dataset，获得对应的值如"sysu"。

    for k, v in dataset_cfg.items():
        # 循环遍历 dataset_cfg 中的所有键值对，添加到cfg中。
        cfg[k] = v
        # 至此cfg包含了所有信息，包括dataset_cfg、strategy_cfg、以及具体的数据集配置文件如SYSU.yml
    cfg.batch_size = cfg.p_size * cfg.k_size
    cfg.freeze()
    # 固定配置信息，防止修改。
    # 开始训练
    train(cfg)
