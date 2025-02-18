# 导入必要的库
import torch
import math
import torch.nn as nn
import numpy as np
import os
from ignite.engine import Engine, Events  # 用于创建训练和评估引擎
from torch.cuda.amp import autocast, GradScaler  # 用于混合精度训练
from torch.autograd import no_grad
from torch.nn import functional as F
import torchvision.transforms as T
# import cv2
from torchvision.io.image import read_image
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms
from grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img  # 用于模型可视化
import copy
from torch.optim.lr_scheduler import LambdaLR


#
def adjust_weight_decay(epoch, initial_weight_decay):
    '''
    定义一个函数来动态调整权重衰减
    :param epoch:
    :param initial_weight_decay:
    :return:
    '''
    if epoch > 15:
        # epoch大于15时，权重衰减为100分之一
        new_weight_decay = initial_weight_decay / 100
    elif epoch > 5 and epoch <= 15:
        new_weight_decay = initial_weight_decay * 1 / 10
    else:
        new_weight_decay = initial_weight_decay
    return new_weight_decay


def create_train_engine(model, optimizer, non_blocking=False):
    '''
    用于创建训练引擎，为Engine类进一步添加自定义方法，随后用于实例化Engine类对象trainer
    '''
    # model是要训练的模型
    device = torch.device("cuda")  # 使用CUDA进行GPU加速
    scaler = GradScaler()  # 初始化 GradScaler，用于混合精度训练

    def _process_func(engine, batch):
        # 一个内部函数，定义了每个批次的训练步骤。接受 Engine 实例和输入批次作为参数。
        # engine是Ignite 的 Engine 实例，包含训练状态（如当前 epoch、iteration 等）。
        model.train()
        epoch = engine.state.epoch
        iteration = engine.state.iteration
        # 拆解传入的 batch，包含三个部分的 batch：total_batch, rgb_batch, ir_batch
        total_batch, rgb_batch, ir_batch = batch
        # 处理每个 batch 数据
        total_data, total_labels, total_cam_ids, total_img_paths, total_img_ids = total_batch
        rgb_data, rgb_labels, rgb_cam_ids, rgb_img_paths, rgb_img_ids = rgb_batch
        ir_data, ir_labels, ir_cam_ids, ir_img_paths, ir_img_ids = ir_batch

        # 将数据移动到 GPU
        total_data = total_data.to(device, non_blocking=non_blocking)
        rgb_data = rgb_data.to(device, non_blocking=non_blocking)
        ir_data = ir_data.to(device, non_blocking=non_blocking)

        total_labels = total_labels.to(device, non_blocking=non_blocking)
        rgb_labels = rgb_labels.to(device, non_blocking=non_blocking)
        ir_labels = ir_labels.to(device, non_blocking=non_blocking)

        # total_cam_ids = total_cam_ids.to(device, non_blocking=non_blocking)
        # rgb_cam_ids = rgb_cam_ids.to(device, non_blocking=non_blocking)
        # ir_cam_ids = ir_cam_ids.to(device, non_blocking=non_blocking)

        # 学习率和权重衰减的warm-up策略
        warmup = False
        if warmup:
            if epoch < 21:
                # 进行warmup，逐渐增加学习率
                warm_iteration = 30 * 213
                lr = 0.00035 * iteration / warm_iteration
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            # 正则化参数warmup
            new_weight_decay = adjust_weight_decay(epoch, 0.5)
            # 调用 adjust_weight_decay 函数，根据当前 epoch 计算新的权重衰减值
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = new_weight_decay
                # 将新的权重衰减值更新到优化器的每个参数组。
        '''
        在前 21 个 epoch 中，按照线性增长策略逐渐提高学习率
        使用 adjust_weight_decay 函数，根据当前的 epoch 计算权重衰减值 (weight_decay)。
        '''
        optimizer.zero_grad()  # 在每次反向传播之前清除之前的梯度

        # with autocast():  # 使用 autocast 启用混合精度
        #     # 对三个分支分别进行前向计算
        #     loss_rgb, metric = model(total_data, rgb_data, ir_data,
        #                          total_labels, rgb_labels, ir_labels,
        #                          total_cam_ids=total_cam_ids.to(device, non_blocking=non_blocking),
        #                          rgb_cam_ids=rgb_cam_ids.to(device, non_blocking=non_blocking),
        #                          ir_cam_ids=ir_cam_ids.to(device, non_blocking=non_blocking))
        #
        # # 反向传播
        # # scaler.scale(loss_total).backward(retain_graph=True)  # 使用 GradScaler 缩放损失值并反向传播
        # scaler.scale(loss_rgb).backward()
        # # scaler.scale(loss_ir).backward()
        # scaler.step(optimizer)  # 使用 GradScaler 更新优化器
        # scaler.update()  # 更新 GradScaler 的缩放因子

        with autocast():  # 使用 autocast 启用混合精度
            loss_student, loss_rgb, loss_ir, metric = model(total_data, rgb_data, ir_data,
                                                            total_labels, rgb_labels, ir_labels,
                                                            total_cam_ids=total_cam_ids.to(device, non_blocking=non_blocking),
                                                            rgb_cam_ids=rgb_cam_ids.to(device, non_blocking=non_blocking),
                                                            ir_cam_ids=ir_cam_ids.to(device, non_blocking=non_blocking))

        if loss_student!=0 and loss_rgb!=0 and loss_ir!=0:
            scaler.scale(loss_student).backward(retain_graph=True)  # 使用 GradScaler 缩放损失值并反向传播
            scaler.scale(loss_rgb).backward(retain_graph=True)
            scaler.scale(loss_ir).backward()
            scaler.step(optimizer)  # 使用 GradScaler 更新优化器
            scaler.update()  # 更新 GradScaler 的缩放因子
        elif loss_student != 0 and loss_rgb!=0 and loss_ir ==0:
            scaler.scale(loss_student).backward(retain_graph=True)
            scaler.scale(loss_rgb).backward()
            scaler.step(optimizer)  # 使用 GradScaler 更新优化器
            scaler.update()  # 更新 GradScaler 的缩放因子
        elif loss_student != 0 and loss_rgb==0 and loss_ir !=0:
            scaler.scale(loss_student).backward(retain_graph=True)
            scaler.scale(loss_ir).backward()
            scaler.step(optimizer)  # 使用 GradScaler 更新优化器
            scaler.update()  # 更新 GradScaler 的缩放因子
        elif loss_student != 0 and loss_rgb==0 and loss_ir ==0:
            scaler.scale(loss_student).backward()
            scaler.step(optimizer)  # 使用 GradScaler 更新优化器
            scaler.update()  # 更新 GradScaler 的缩放因子
        elif loss_student == 0 and loss_rgb!=0 and loss_ir ==0:
            scaler.scale(loss_rgb).backward()
            scaler.step(optimizer)  # 使用 GradScaler 更新优化器
            scaler.update()  # 更新 GradScaler 的缩放因子
        elif loss_student == 0 and loss_rgb==0 and loss_ir !=0:
            scaler.scale(loss_ir).backward()
            scaler.step(optimizer)  # 使用 GradScaler 更新优化器
            scaler.update()  # 更新 GradScaler 的缩放因子
        # 创建一个包含非零损失项的列表
        # losses = []
        # if loss_student != 0:
        #     losses.append(loss_student)
        # if loss_rgb != 0:
        #     losses.append(loss_rgb)
        # if loss_ir != 0:
        #     losses.append(loss_ir)
        #
        # # 如果有损失项，进行反向传播和优化
        # if losses:
        #     # 反向传播
        #     for idx, loss in enumerate(losses):
        #         retain = True if idx == 0 else False  # 只有第一次反向传播需要retain_graph=True
        #         scaler.scale(loss).backward(retain_graph=retain)
        #
        #     # 更新优化器
        #     scaler.step(optimizer)
        #     scaler.update()  # 更新 GradScaler 的缩放因子

        return metric

    return Engine(_process_func)
    # Engine类可以用于训练和评估模型，需要提供参数_process_func，定义了每个训练步骤中需要执行的操作


# def create_eval_engine(model, non_blocking=False):
#     device = torch.device("cuda", torch.cuda.current_device())
#
#     # 提取公共的转换到设备的代码
#     def move_to_device(data, non_blocking):
#         return data.to(device, non_blocking=non_blocking)
#
#     # 统一的返回构建函数
#     def build_return(logit_student, feat_total, total_labels, total_cam_ids, total_img_paths,
#                      feat_query, labels_query, cam_ids_query, img_paths_query,
#                      feat_gallery, labels_gallery, cam_ids_gallery, img_paths_gallery, is_rgb):
#         return (logit_student.data.float().cpu(),
#                 feat_total.data.float().cpu(), total_labels, total_cam_ids, np.array(total_img_paths),
#                 feat_query.data.float().cpu(), labels_query, cam_ids_query, np.array(img_paths_query),
#                 feat_gallery.data.float().cpu(), labels_gallery, cam_ids_gallery, np.array(img_paths_gallery),
#                 is_rgb)
#
#
#
#     def _process_func(engine, batch):
#         model.eval()  # 设置模型为评估模式
#
#         # 解包 batch 中的三个部分：total_batch, rgb_batch_query, rgb_batch_gallery
#         total_batch, query_batch, gallery_batch = batch
#         total_data, total_labels, total_cam_ids, total_img_paths, total_img_ids = total_batch
#
#         # 判断模态
#         is_rgb = total_cam_ids[0] in [1, 2, 4, 5]
#
#         # 转移数据到设备
#         total_data = move_to_device(total_data, non_blocking)
#
#         if is_rgb:
#             rgb_data_query, rgb_labels_query, rgb_cam_ids_query, rgb_img_paths_query, rgb_img_ids_query = query_batch
#             rgb_data_gallery, rgb_labels_gallery, rgb_cam_ids_gallery, rgb_img_paths_gallery, rgb_img_ids_gallery = gallery_batch
#
#             rgb_data_query = move_to_device(rgb_data_query, non_blocking)
#             rgb_data_gallery = move_to_device(rgb_data_gallery, non_blocking)
#
#             with torch.no_grad():
#                 logit_student, feat_total, feat_rgb_query, feat_rgb_gallery = model(total_data,
#                                                                                     rgb_data_query,
#                                                                                     rgb_data_gallery,
#                                                                                     total_labels,
#                                                                                     rgb_labels_query,
#                                                                                     rgb_labels_gallery,
#                                                                                     rgb_test=True, ir_test=False)
#                 return build_return(logit_student, feat_total, total_labels, total_cam_ids, total_img_paths,
#                                     feat_rgb_query, rgb_labels_query, rgb_cam_ids_query, rgb_img_paths_query,
#                                     feat_rgb_gallery, rgb_labels_gallery, rgb_cam_ids_gallery, rgb_img_paths_gallery,
#                                     is_rgb, logit_student)
#         else:
#             ir_data_query, ir_labels_query, ir_cam_ids_query, ir_img_paths_query, ir_img_ids_query = query_batch
#             ir_data_gallery, ir_labels_gallery, ir_cam_ids_gallery, ir_img_paths_gallery, ir_img_ids_gallery = gallery_batch
#
#             ir_data_query = move_to_device(ir_data_query, non_blocking)
#             ir_data_gallery = move_to_device(ir_data_gallery, non_blocking)
#
#             with torch.no_grad():
#                 logit_student, feat_total, feat_ir_query, feat_ir_gallery = model(total_data,
#                                                                                   ir_data_query,
#                                                                                   ir_data_gallery,
#                                                                                   total_labels,
#                                                                                   ir_labels_query,
#                                                                                   ir_labels_gallery,
#                                                                                   rgb_test=False, ir_test=True)
#
#                 return build_return(feat_total, total_labels, total_cam_ids, total_img_paths,
#                                     feat_ir_query, ir_labels_query, ir_cam_ids_query, ir_img_paths_query,
#                                     feat_ir_gallery, ir_labels_gallery, ir_cam_ids_gallery, ir_img_paths_gallery,
#                                     is_rgb, logit_student)
#
#     engine = Engine(_process_func)



# 创建评估引擎
def create_eval_engine(model, non_blocking=False):

    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()  # 设置模型为评估模式

        # 解包 batch 中的三个部分：total_batch, rgb_batch_query, rgb_batch_gallery
        total_batch, query_batch, gallery_batch = batch
        total_data, total_labels, total_cam_ids, total_img_paths, total_img_ids = total_batch
        is_rgb = True
        if total_cam_ids[0] == 1 or total_cam_ids[0] == 2 or total_cam_ids[0] == 4 or total_cam_ids[0] == 5:  # RGB 模态
            is_rgb = True
        else:
            is_rgb = False

        if is_rgb:
            rgb_data_query, rgb_labels_query, rgb_cam_ids_query, rgb_img_paths_query, rgb_img_ids_query = query_batch
            rgb_data_gallery, rgb_labels_gallery, rgb_cam_ids_gallery, rgb_img_paths_gallery, rgb_img_ids_gallery = gallery_batch

            total_data = total_data.to(device, non_blocking=non_blocking)
            rgb_data_query = rgb_data_query.to(device, non_blocking=non_blocking)
            rgb_data_gallery = rgb_data_gallery.to(device, non_blocking=non_blocking)
        else:
            ir_data_query, ir_labels_query, ir_cam_ids_query, ir_img_paths_query, ir_img_ids_query = query_batch
            ir_data_gallery, ir_labels_gallery, ir_cam_ids_gallery, ir_img_paths_gallery, ir_img_ids_gallery = gallery_batch

            total_data = total_data.to(device, non_blocking=non_blocking)
            ir_data_query = ir_data_query.to(device, non_blocking=non_blocking)
            ir_data_gallery = ir_data_gallery.to(device, non_blocking=non_blocking)

        with no_grad():
            if is_rgb:
                logit_student, feat_total, feat_rgb_query, feat_rgb_gallery = model(total_data,
                                                                                    rgb_data_query,
                                                                                    rgb_data_gallery,
                                                                                    total_labels,
                                                                                    rgb_labels_query,
                                                                                    rgb_labels_gallery,
                                                                                    rgb_test=True, ir_test=False)
                return (feat_total.data.float().cpu(), total_labels, total_cam_ids, np.array(total_img_paths),
                        feat_rgb_query.data.float().cpu(), rgb_labels_query, rgb_cam_ids_query,
                        np.array(rgb_img_paths_query),
                        feat_rgb_gallery.data.float().cpu(), rgb_labels_gallery, rgb_cam_ids_gallery,
                        np.array(rgb_img_paths_gallery),
                        is_rgb, logit_student.data.float().cpu())
            else:
                logit_student, feat_total, feat_ir_query, feat_ir_gallery = model(total_data,
                                                                                  ir_data_query,
                                                                                  ir_data_gallery,
                                                                                  total_labels,
                                                                                  ir_labels_query,
                                                                                  ir_labels_gallery,
                                                                                  rgb_test=False, ir_test=True)

                return (feat_total.data.float().cpu(), total_labels, total_cam_ids, np.array(total_img_paths),
                        feat_ir_query.data.float().cpu(), ir_labels_query, ir_cam_ids_query,
                        np.array(ir_img_paths_query),
                        feat_ir_gallery.data.float().cpu(), ir_labels_gallery, ir_cam_ids_gallery,
                        np.array(ir_img_paths_gallery),
                        is_rgb, logit_student.data.float().cpu())

    engine = Engine(_process_func)

    # 在每个epoch开始时清除数据
    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # 初始化或清除特征列表、ID列表、相机列表和图像路径列表
        for attr in ["total_feat_list", "total_id_list", "total_cam_list", "total_img_path_list",
                     "rgb_feat_list_query", "rgb_id_list_query", "rgb_cam_list_query", "rgb_img_path_list_query",
                     "rgb_feat_list_gallery", "rgb_id_list_gallery", "rgb_cam_list_gallery", "rgb_img_path_list_gallery",
                     "ir_feat_list_query", "ir_id_list_query", "ir_cam_list_query", "ir_img_path_list_query",
                     "ir_feat_list_gallery", "ir_id_list_gallery", "ir_cam_list_gallery", "ir_img_path_list_gallery",
                     "total_logit_list"]:
            if not hasattr(engine.state, attr):
                setattr(engine.state, attr, [])
            else:
                getattr(engine.state, attr).clear()

    # 在每次迭代完成后存储数据
    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        # 公共操作部分
        def append_data(feat_list, id_list, cam_list, img_path_list, output_idx):
            feat_list.append(engine.state.output[output_idx])
            id_list.append(engine.state.output[output_idx + 1])
            cam_list.append(engine.state.output[output_idx + 2])
            img_path_list.append(engine.state.output[output_idx + 3])

        # 存储公共数据
        engine.state.total_feat_list.append(engine.state.output[0])
        engine.state.total_id_list.append(engine.state.output[1])
        engine.state.total_cam_list.append(engine.state.output[2])
        engine.state.total_img_path_list.append(engine.state.output[3])
        engine.state.total_logit_list.append(engine.state.output[13])

        # 根据模态选择存储数据
        if engine.state.output[12]:  # RGB 模态
            append_data(engine.state.rgb_feat_list_query, engine.state.rgb_id_list_query,
                        engine.state.rgb_cam_list_query, engine.state.rgb_img_path_list_query, 4)
            append_data(engine.state.rgb_feat_list_gallery, engine.state.rgb_id_list_gallery,
                        engine.state.rgb_cam_list_gallery, engine.state.rgb_img_path_list_gallery, 8)
        else:  # IR 模态
            append_data(engine.state.ir_feat_list_query, engine.state.ir_id_list_query, engine.state.ir_cam_list_query,
                        engine.state.ir_img_path_list_query, 4)
            append_data(engine.state.ir_feat_list_gallery, engine.state.ir_id_list_gallery,
                        engine.state.ir_cam_list_gallery, engine.state.ir_img_path_list_gallery, 8)

    return engine