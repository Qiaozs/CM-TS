import logging
import os
import numpy as np
import torch
import scipy.io as sio
import torch.nn as nn

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from tqdm import tqdm  # tqdm 是进度条工具

from .engine import create_eval_engine
from .engine import create_train_engine
from .metric import AutoKVMetric
from utils.eval_regdb import eval_regdb
from utils.eval_regdb import eval_regdb_for_teacher
from utils.eval_sysu import eval_sysu
from utils.eval_sysu import eval_sysu_for_teacher
from utils.eval_llcm import eval_llcm
from utils.calc_acc import calc_acc
from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg



def get_test_feats(Dataloader, evaluator):
    evaluator.run(Dataloader)
    # 通过 evaluator 引擎运行 query_loader 数据集。
    # 会计算并存储模型的特征输出和其他信息（如 ID、摄像头、图像路径等）在 evaluator.state 对象中。
    feats = torch.cat(evaluator.state.total_feat_list, dim=0)  # 将 evaluator.state.feat_list 中的张量按维度 0 拼接，得到查询数据的完整特征。
    ids = torch.cat(evaluator.state.total_id_list, dim=0)  # 拼接查询数据的 ID。
    cams = torch.cat(evaluator.state.total_cam_list, dim=0).numpy()  # 拼接查询数据的摄像头 ID。
    img_paths = np.concatenate(evaluator.state.total_img_path_list, axis=0)  # 拼接查询数据的图像路径列表。
    logit = torch.cat(evaluator.state.total_logit_list, dim=0)
    if evaluator.state.output[12]:
        rgb_feats_query = torch.cat(evaluator.state.rgb_feat_list_query, dim=0)
        rgb_ids_query = torch.cat(evaluator.state.rgb_id_list_query, dim=0).numpy()
        rgb_cams_query = torch.cat(evaluator.state.rgb_cam_list_query, dim=0).numpy()
        rgb_img_paths_query = np.concatenate(evaluator.state.rgb_img_path_list_query, axis=0)
        rgb_feats_gallery = torch.cat(evaluator.state.rgb_feat_list_gallery, dim=0)
        rgb_ids_gallery = torch.cat(evaluator.state.rgb_id_list_gallery, dim=0).numpy()
        rgb_cams_gallery = torch.cat(evaluator.state.rgb_cam_list_gallery, dim=0).numpy()
        rgb_img_paths_gallery = np.concatenate(evaluator.state.rgb_img_path_list_gallery, axis=0)
        # 清除 evaluator 的状态信息：
        evaluator.state.total_feat_list.clear()
        evaluator.state.total_id_list.clear()
        evaluator.state.total_cam_list.clear()
        evaluator.state.total_img_path_list.clear()
        evaluator.state.rgb_feat_list_query.clear()
        evaluator.state.rgb_id_list_query.clear()
        evaluator.state.rgb_cam_list_query.clear()
        evaluator.state.rgb_img_path_list_query.clear()
        evaluator.state.rgb_feat_list_gallery.clear()
        evaluator.state.rgb_id_list_gallery.clear()
        evaluator.state.rgb_cam_list_gallery.clear()
        evaluator.state.rgb_img_path_list_gallery.clear()

        evaluator.state.total_logit_list.clear()

        return (logit, feats, ids, cams, img_paths,
                rgb_feats_query, rgb_ids_query, rgb_cams_query, rgb_img_paths_query,
                rgb_feats_gallery, rgb_ids_gallery, rgb_cams_gallery, rgb_img_paths_gallery)
    else:
        ir_feats_query = torch.cat(evaluator.state.ir_feat_list_query,
                                    dim=0)  # 将 evaluator.state.feat_list 中的张量按维度 0 拼接，得到查询数据的完整特征。
        ir_ids_query = torch.cat(evaluator.state.ir_id_list_query, dim=0).numpy()  # 拼接查询数据的 ID。
        ir_cams_query = torch.cat(evaluator.state.ir_cam_list_query, dim=0).numpy()  # 拼接查询数据的摄像头 ID。
        ir_img_paths_query = np.concatenate(evaluator.state.ir_img_path_list_query, axis=0)  # 拼接查询数据的图像路径列表。
        ir_feats_gallery = torch.cat(evaluator.state.ir_feat_list_gallery,
                                      dim=0)  # 将 evaluator.state.feat_list 中的张量按维度 0 拼接，得到查询数据的完整特征。
        ir_ids_gallery = torch.cat(evaluator.state.ir_id_list_gallery, dim=0).numpy()  # 拼接查询数据的 ID。
        ir_cams_gallery = torch.cat(evaluator.state.ir_cam_list_gallery, dim=0).numpy()  # 拼接查询数据的摄像头 ID。
        ir_img_paths_gallery = np.concatenate(evaluator.state.ir_img_path_list_gallery, axis=0)  # 拼接查询数据的图像路径列表。
        # 清除 evaluator 的状态信息：
        evaluator.state.total_feat_list.clear()
        evaluator.state.total_id_list.clear()
        evaluator.state.total_cam_list.clear()
        evaluator.state.total_img_path_list.clear()
        evaluator.state.ir_feat_list_query.clear()
        evaluator.state.ir_id_list_query.clear()
        evaluator.state.ir_cam_list_query.clear()
        evaluator.state.ir_img_path_list_query.clear()
        evaluator.state.ir_feat_list_gallery.clear()
        evaluator.state.ir_id_list_gallery.clear()
        evaluator.state.ir_cam_list_gallery.clear()
        evaluator.state.ir_img_path_list_gallery.clear()

        evaluator.state.total_logit_list.clear()
        return (logit, feats, ids, cams, img_paths,
                ir_feats_query, ir_ids_query, ir_cams_query, ir_img_paths_query,
                ir_feats_gallery, ir_ids_gallery, ir_cams_gallery, ir_img_paths_gallery)


loss_fn = nn.CrossEntropyLoss(ignore_index=-1)


def get_trainer(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False,
                log_period=10,
                save_dir="checkpoints", prefix="model", gallery_loader=None, query_loader=None,
                eval_interval=None, start_eval=None, rerank=False):
    '''
    使用create_train_engine()函数实例化一个Engine对象trainer
    使用create_eval_engine()函数实例化一个Engine对象evaluator，在create_eval_engine()中已经定义了不同事件及操作
    为trainer注册各个事件以实现在不同的训练过程中执行不同的操作：
       1. Events.STARTED
       2. Events.COMPLETED：释放显存，调用evaluator进行评估
       3. Events.EPOCH_STARTED： 在一个epoch开始时，执行权重动态更新策略
       4. Events.EPOCH_COMPLETED：在一个 epoch 结束后，触发检查点保存操作，学习率更新操作，评估操作
       5. Events.ITERATION_COMPLETED：在每一批次训练结束后，根据批次数输出日志信息

    Args:
        dataset: 字符串
        model: 模型
        optimizer: 优化器
        lr_scheduler: 学习率策略
        logger: 日志记录器
        writer: 日志写入器
        non_blocking=False: 以非阻塞方式将数据传输到 GPU。
        log_period: 默认为 10。用来设置日志输出的间隔（例如每 10 个步骤记录一次日志）。
        save_dir:
        prefix:
        gallery_loader: 在train.py内指定，使用get_test_loader()函数得到
        query_loader: 在train.py内指定，使用get_test_loader()函数得到
        eval_interval:
        start_eval:
        rerank:

    Returns:
    '''
    # 检查并初始化日志记录器
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    # trainer
    trainer = create_train_engine(model, optimizer, non_blocking)
    print("创建trainer用于训练")
    # create_train_engine返回一个Engine类，其规定了训练过程中每一个批次的训练步骤。
    # trainer为一个Engine类对象

    setattr(trainer, "rerank", rerank)  # setattr 函数将 rerank 属性动态地添加到 trainer 对象中

    # checkpoint handler
    # handler = ModelCheckpoint(save_dir, prefix, n_saved=3, create_dir=True,
    #                           save_as_state_dict=True, require_empty=False)
    handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=3, create_dir=True,
                              save_as_state_dict=True, require_empty=False)
    '''
    ModelCheckpoint 对象，用于定期保存模型检查点。
    save_dir: 指定保存检查点文件的目录
    eval_interval：每隔eval_interval个epoch间隔保存一次模型
    n_saved: 最多保存n_saved个检查点
    create_dir: 如果save_dir不存在，是否创建该目录
    save_as_state_dict: 是否将模型转换为 state_dict 并保存 
    require_empty: 如果save_dir存在，是否检查是否有任何检查点
    '''
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model, "optimizer": optimizer})
    # 将 检查点处理器handler 绑定到 trainer 的 EPOCH_COMPLETED 事件上。当一个 epoch 完成时，触发检查点保存操作。
    # model指定要保存的对象

    # metric
    timer = Timer(average=True)  # timer用于测量时间
    rank = False  # 用于指定

    kv_metric = AutoKVMetric()  # AutoKVMetric 一个度量工具，用于自动计算和记录关键值指标（Key-Value Metrics）

    # evaluator
    evaluator = None  # 默认情况下没有定义评估器。
    # 检测eval_interval和start_eval是否为整数。
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        evaluator = create_eval_engine(model, non_blocking)
        print("创建evaluator用于测试")
        # 如果检测间隔 eval_interval 是整数 & gallery_loader 和 query_loader 非空，则使用create_eval_engine()创建一个评估引擎。

    # 向trainer这个Engine类对象中进一步添加各个事件及触发动作
    # 一个装饰器，用于注册 train_start 函数为 trainer 对象的事件处理器。
    # 当Events.STARTED事件触发，执行函数train_start
    @trainer.on(Events.STARTED)
    def train_start(engine):
        # 接受参数engine，为一个trainer的实例
        setattr(engine.state, "best_Total_rank1", 0.0)
        setattr(engine.state, "best_RGB_rank1", 0.0)
        setattr(engine.state, "best_IR_rank1", 0.0)
        setattr(engine.state, "best_RGB_acc", 0.0)
        setattr(engine.state, "best_IR_acc", 0.0)
        # 将将 best_rank1 设置为 0.0，并附加到 engine.state 对象中。
        # best_rank1 用于存储模型在评估任务的最佳排名指标，

    @trainer.on(Events.EPOCH_STARTED)
    # 每当训练循环进入新的 epoch 时，触发该事件。
    def epoch_started_callback(engine):

        epoch = engine.state.epoch  # 获取当前 epoch 的索引
        if model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_
            # model.update_rate 模型的动态更新率参数
            # 随着 epoch 的增加，update_rate 会逐渐减小
        kv_metric.reset()
        timer.reset()  # 重置记录的指标和计时器
    @trainer.on(Events.ITERATION_COMPLETED)
    # 每个批次的数据训练完成后都会调用该函数。
    def iteration_complete_callback(engine):
        timer.step()  # 更新计时器，记录当前批次所用的时间。

        kv_metric.update(engine.state.output)

        # 计算迭代的相关信息
        epoch = engine.state.epoch
        iteration = engine.state.iteration  # 从训练开始到当前的总迭代次数。
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)  # 计算当前 epoch 内的迭代编号。

        # 记录训练速度
        if iter_in_epoch % log_period == 0 and iter_in_epoch > 0:
            # 每 log_period 个批次记录一次日志。
            batch_size_total = engine.state.batch[0][0].size(0)  # engine.state.batch[0].size(0)计算当前批次样本数量
            batch_size_rgb = engine.state.batch[1][0].size(0)  # RGB batch
            batch_size_ir = engine.state.batch[2][0].size(0)  # IR batch
            speed_total = batch_size_total / timer.value()
            speed_rgb = batch_size_rgb / timer.value()
            speed_ir = batch_size_ir / timer.value()

            #
            msg = " Epoch[%d] Total_Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed_total)
            msg += " Epoch[%d] RGB_Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed_rgb)
            msg += " Epoch[%d] IR_Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed_ir)
            # 构建日志，记录当前 epoch、第几个批次以及训练速度。

            metric_dict = kv_metric.compute()  # 计算累计的指标值，返回一个字典。
            # 输出日志信息
            if logger is not None:
                for k in sorted(metric_dict.keys()):  # 遍历所有的指标，将其附加到日志消息中。
                    msg += "\t%s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)
                logger.info(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch
        if lr_scheduler is not None:
            lr_scheduler.step()
            # 如果没有提供学习率调度器，则调用lr_scheduler.step()方法，不过在train.py中指定了

        if epoch % eval_interval == 0:  # 每当到达记录间隔，将一条日志记录到日志文件或控制台，表示模型在当前 epoch 已保存。
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0 and epoch > start_eval:
            # 如果存在evaluator且当前epoch达到设置的评估条件，开始评估
            torch.cuda.empty_cache()

            logger.info("训练结束开始加载query")
            (ir_logit, q_feats, q_ids, q_cams, q_img_paths,
             ir_feats_query, ir_ids_query, ir_cams_query, ir_img_paths_query,
             ir_feats_gallery, ir_ids_gallery, ir_cams_gallery, ir_img_paths_gallery) = get_test_feats(query_loader,
                                                                                                       evaluator)

            # extract gallery feature
            logger.info("训练结束开始加载gallery")
            (rgb_logit, g_feats, g_ids, g_cams, g_img_paths,
             rgb_feats_query, rgb_ids_query, rgb_cams_query, rgb_img_paths_query,
             rgb_feats_gallery, rgb_ids_gallery, rgb_cams_gallery, rgb_img_paths_gallery) = get_test_feats(
                gallery_loader,
                evaluator)

            if dataset == 'sysu':
                path = os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat')
                standard_path = os.path.normpath(path)
                perm = sio.loadmat(standard_path)['rand_perm_cam']
                print('SYSU测试方式：IR单模态分类识别，基于RGB_Teacher模型，测试结果如下：')
                print("rgb_acc:", calc_acc(rgb_logit.data, g_ids))
                print('SYSU测试方式：IR单模态分类识别，基于IR_Teacher模型，测试结果如下：')
                print("ir_acc:", calc_acc(ir_logit.data, q_ids))

                print('SYSU测试方式：RGB单模态内重识别，基于RGB_Teacher模型，测试结果如下：')
                rgb_mAP, rgb_r1, rgb_r5, _, _ = eval_sysu_for_teacher(rgb_feats_query, rgb_ids_query, rgb_cams_query,
                                                                      rgb_feats_gallery, rgb_ids_gallery,
                                                                      rgb_cams_gallery, rgb_img_paths_gallery, perm,
                                                                      mode='all', mode1='rgb', num_shots=1, rerank=rank)


                print('SYSU测试方式：IR单模态内重识别，基于IR_Teacher模型，测试结果如下：')
                ir_mAP, ir_r1, ir_r5, _, _ = eval_sysu_for_teacher(ir_feats_query, ir_ids_query, ir_cams_query,
                                                                   ir_feats_gallery, ir_ids_gallery, ir_cams_gallery,
                                                                   ir_img_paths_gallery, perm,
                                                                   mode='all', mode1='ir', num_shots=1, rerank=rank)

                print('SYSU测试方式：跨模态重识别，基于Student模型，测试结果如下：')
                total_mAP, total_r1, total_r5, _, _ = eval_sysu(q_feats, q_ids.numpy(), q_cams, g_feats, g_ids.numpy(), g_cams, g_img_paths,
                                                                perm, mode='all', num_shots=1, rerank=rank)

                # 调用eval_sysu()
            if dataset == 'regdb':
                print('RegDB测试方式：IR单模态分类识别，基于RGB_Teacher模型，测试结果如下：')
                print("rgb_acc:", calc_acc(rgb_logit, g_ids))
                print('RegDB测试方式：IR单模态分类识别，基于IR_Teacher模型，测试结果如下：')
                print("ir_acc:", calc_acc(ir_logit, q_ids))
                print('RegDB测试方式：RGB单模态内搜索，基于RGB_Teacher模型，测试结果如下：')
                rgb_mAP, rgb_r1, rgb_r5, _, _ = eval_regdb_for_teacher(rgb_feats_query, rgb_ids_query, rgb_cams_query,
                                                                       rgb_feats_gallery, rgb_ids_gallery,
                                                                       rgb_cams_gallery,
                                                                       rerank=engine.rerank)
                print('RegDB测试方式：IR单模态内搜索，基于IR_Teacher模型，测试结果如下：')
                ir_mAP, ir_r1, ir_r5, _, _ = eval_regdb_for_teacher(ir_feats_query, ir_ids_query, ir_cams_query,
                                                                    ir_feats_gallery, ir_ids_gallery, ir_cams_gallery,
                                                                    rerank=engine.rerank)
                print('RegDB测试方式：infrared to visible，基于Student模型，测试结果如下：')
                total_mAP, total_r1, total_r5, _, _ = eval_regdb(q_feats, q_ids.numpy(), q_cams, g_feats, g_ids.numpy(), g_cams,
                                                                 rerank=engine.rerank)
                print('RegDB测试方式：visible to infrared，基于Student模型，测试结果如下：')
                total_mAP, total_r1_, total_r5, _, _ = eval_regdb(g_feats, g_ids.numpy(), g_cams, q_feats, q_ids.numpy(), q_cams,
                                                                  rerank=engine.rerank)

                total_r1 = (total_r1 + total_r1_) / 2

            if rgb_r1 > engine.state.best_RGB_rank1:
                engine.state.best_RGB_rank1 = rgb_r1
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))
                # 如果 r1 大于当前历史最佳准确率，则更新
                logger.info(f"最新的 RGB_Rank-1 准确率 : {rgb_r1:.2f}% (model saved to {save_dir})")

            if ir_r1 > engine.state.best_IR_rank1:
                engine.state.best_IR_rank1 = ir_r1
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))
                # 如果 r1 大于当前历史最佳准确率，则更新
                logger.info(f"最新的 IR_Rank-1 准确率: {ir_r1:.2f}% (model saved to {save_dir})")

            if total_r1 > engine.state.best_Total_rank1:
                engine.state.best_Total_rank1 = total_r1
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))
                # 如果 r1 大于当前历史最佳准确率，则更新
                logger.info(f"最新的 Total_Rank-1 准确率: {total_r1:.2f}% (model saved to {save_dir})")

            if writer is not None:
                writer.add_scalar('eval/total_mAP', total_mAP, epoch)
                writer.add_scalar('eval/total_r1', total_r1, epoch)
                writer.add_scalar('eval/total_r5', total_r5, epoch)
                writer.add_scalar('eval/rgb_mAP', rgb_mAP, epoch)
                writer.add_scalar('eval/rgb_r1', rgb_r1, epoch)
                writer.add_scalar('eval/rgb_r5', rgb_r5, epoch)
                writer.add_scalar('eval/ir_mAP', ir_mAP, epoch)
                writer.add_scalar('eval/ir_r1', ir_r1, epoch)
                writer.add_scalar('eval/ir_r5', ir_r5, epoch)

            del rgb_logit, ir_logit
            del q_feats, q_ids, q_cams
            del g_feats, g_ids, g_cams
            del rgb_feats_query, rgb_ids_query, rgb_cams_query, rgb_img_paths_query
            del rgb_feats_gallery, rgb_ids_gallery, rgb_cams_gallery, rgb_img_paths_gallery
            del ir_feats_query, ir_ids_query, ir_cams_query, ir_img_paths_query
            del ir_feats_gallery, ir_ids_gallery, ir_cams_gallery, ir_img_paths_gallery

            # progress_bar.close()

            torch.cuda.empty_cache()
    @trainer.on(Events.COMPLETED)
    def train_completed(engine):
        torch.cuda.empty_cache()  # 手动释放未被使用的显存

        # extract query feature
        logger.info("训练结束开始加载query")
        (ir_logit, q_feats, q_ids, q_cams, q_img_paths,
         ir_feats_query, ir_ids_query, ir_cams_query, ir_img_paths_query,
         ir_feats_gallery, ir_ids_gallery, ir_cams_gallery, ir_img_paths_gallery) = get_test_feats(query_loader,
                                                                                                   evaluator)

        # extract gallery feature
        logger.info("训练结束开始加载gallery")
        (rgb_logit, g_feats, g_ids, g_cams, g_img_paths,
         rgb_feats_query, rgb_ids_query, rgb_cams_query, rgb_img_paths_query,
         rgb_feats_gallery, rgb_ids_gallery, rgb_cams_gallery, rgb_img_paths_gallery) = get_test_feats(
            gallery_loader,
            evaluator)
        if dataset == 'sysu':
            path = os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat')
            standard_path = os.path.normpath(path)
            perm = sio.loadmat(standard_path)['rand_perm_cam']

            print('SYSU测试方式：RGB单模态内搜索，基于RGB_Teacher模型，测试结果如下：')
            eval_sysu_for_teacher(rgb_feats_query, rgb_ids_query, rgb_cams_query,
                                  rgb_feats_gallery, rgb_ids_gallery, rgb_cams_gallery, rgb_img_paths_gallery, perm,
                                  mode='all', mode1='rgb', num_shots=1, rerank=rank)
            print('SYSU测试方式：IR单模态内搜索，基于IR_Teacher模型，测试结果如下：')
            eval_sysu_for_teacher(ir_feats_query, ir_ids_query, ir_cams_query,
                                  ir_feats_gallery, ir_ids_gallery, ir_cams_gallery, ir_img_paths_gallery, perm,
                                  mode='all', mode1='ir', num_shots=1, rerank=rank)
            print('SYSU测试方式：跨模态搜索，基于Student模型，测试结果如下：')
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm,
                      mode='all', num_shots=1, rerank=rank)
            # print('SYSU测试方式：visible to infrared，基于Student模型，测试结果如下：')
            # eval_sysu(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, perm,
            #                                                  mode='all', num_shots=1, rerank=rank)
            # mAP, r1, r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=rank)

            # 调用eval_sysu()
        if dataset == 'regdb':
            print('RegDB测试方式：IR单模态分类识别，基于RGB_Teacher模型，测试结果如下：')
            print("rgb_acc:", calc_acc(rgb_logit.data, g_ids))
            print('RegDB测试方式：IR单模态分类识别，基于IR_Teacher模型，测试结果如下：')
            print("ir_acc:", calc_acc(ir_logit.data, q_ids))

            print('RegDB测试方式：RGB单模态内搜索，基于RGB_Teacher模型，测试结果如下：')
            eval_regdb_for_teacher(rgb_feats_query, rgb_ids_query, rgb_cams_query,
                                                                   rgb_feats_gallery, rgb_ids_gallery,
                                                                   rgb_cams_gallery,
                                                                   rerank=engine.rerank)
            print('RegDB测试方式：IR单模态内搜索，基于IR_Teacher模型，测试结果如下：')
            eval_regdb_for_teacher(ir_feats_query, ir_ids_query, ir_cams_query,
                                                                ir_feats_gallery, ir_ids_gallery, ir_cams_gallery,
                                                                rerank=engine.rerank)


            print('RegDB测试方式：infrared to visible，基于Student模型，测试结果如下：')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams,
                                                             rerank=engine.rerank)
            print('RegDB测试方式：visible to infrared，基于Student模型，测试结果如下：')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams,
                                                              rerank=engine.rerank)

        del q_feats, q_ids, q_cams
        del g_feats, g_ids, g_cams
        del rgb_feats_query, rgb_ids_query, rgb_cams_query, rgb_img_paths_query
        del rgb_feats_gallery, rgb_ids_gallery, rgb_cams_gallery, rgb_img_paths_gallery
        del ir_feats_query, ir_ids_query, ir_cams_query, ir_img_paths_query
        del ir_feats_gallery, ir_ids_gallery, ir_cams_gallery, ir_img_paths_gallery

        torch.cuda.empty_cache()
        # 手动释放未被使用的显存

    return trainer
