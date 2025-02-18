import os
import logging
import numpy as np
import torch
# from sklearn.preprocessing import normalize
from torch.nn import functional as F
from .rerank import re_ranking, pairwise_distance

# 获取图库样本的名称，用于结果展示
def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()  # 获取摄像头 perm 中对应的索引
        for i in ids:
            # 对每个 ID，获取对应的实例（trial_id）并生成名称，格式： cam{cam}/{ID}/{instance_id}
            instance_id = cam_perm[i - 1][trial_id][:num_shots]  # 获取对应 trial_id 的实例
            names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])
    return names

# 获取数组中的唯一元素
def get_unique(array):
    _, idx = np.unique(array, return_index=True)  # 获取唯一元素及其索引
    return array[np.sort(idx)]  # 按索引排序并返回唯一元素

# 计算 Cumulative Matching Characteristics (CMC)
def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    # 获取图库中唯一 ID 的数量
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))  # 初始化匹配计数

    result = gallery_ids[sorted_indices]  # 获取排序后的图库 ID
    cam_locations_result = gallery_cam_ids[sorted_indices]  # 获取排序后的图库相机 ID

    valid_probe_sample_count = 0  # 有效查询样本数量

    for probe_index in range(sorted_indices.shape[0]):
        # 对于每个查询样本（probe_index），从 result 中取出排序后的图库 ID
        # 去除与查询样本在同一相机中的图库样本
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # 移除标签结果中的 -1 项
        result_i = np.array([i for i in result_i if i != -1])  # 排除同相机的样本

        # 对结果去重
        result_i_unique = get_unique(result_i)

        # 判断查询样本与图库中样本是否匹配
        match_i = np.equal(result_i_unique, query_ids[probe_index])  # match_i 是一个布尔数组，表示排序后的图库样本是否与查询的 ID 匹配。

        if np.sum(match_i) != 0:  # 如果图库中有匹配的样本
            valid_probe_sample_count += 1
            match_counter += match_i  # 统计匹配的图库样本数量

    rank = match_counter / valid_probe_sample_count  # 计算每个位置的排名精度
    cmc = np.cumsum(rank)  # 计算累计精度
    return cmc

def get_cmc_for_teacher(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    # 获取图库中唯一 ID 的数量
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))  # 初始化匹配计数

    result = gallery_ids[sorted_indices]  # 获取排序后的图库 ID
    cam_locations_result = gallery_cam_ids[sorted_indices]  # 获取排序后的图库相机 ID

    valid_probe_sample_count = 0  # 有效查询样本数量

    for probe_index in range(sorted_indices.shape[0]):
        # 对于每个查询样本（probe_index），从 result 中取出排序后的图库 ID
        result_i = result[probe_index, :]
        # 对结果去重
        result_i_unique = get_unique(result_i)
        # 判断查询样本与图库中样本是否匹配
        match_i = np.equal(result_i_unique, query_ids[probe_index])  # match_i 是一个布尔数组，表示排序后的图库样本是否与查询的 ID 匹配。
        # print("match_i:", match_i)
        if np.sum(match_i) != 0:  # 如果图库中有匹配的样本
            valid_probe_sample_count += 1
            match_counter += match_i  # 统计匹配的图库样本数量
        # print("match_counter:",match_counter)

    rank = match_counter / valid_probe_sample_count  # 计算每个位置的排名精度
    cmc = np.cumsum(rank)  # 计算累计精度
    return cmc
# 计算均值平均精度 (mAP)
def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]
    # print(f"Shape of result: {result.shape}")
    # print(f"Contents of result: \n{result}")
    valid_probe_sample_count = 0  # 有效查询样本数量
    avg_precision_sum = 0  # 平均精度和

    for probe_index in range(sorted_indices.shape[0]):
        # 去除与查询样本在同一相机中的图库样本
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # 移除标签结果中的 -1 项
        result_i = np.array([i for i in result_i if i != -1])

        # 判断查询样本与图库中样本是否匹配
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)  # 统计匹配数量

        if true_match_count != 0:  # 如果图库中有匹配的样本
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            # 计算平均精度 (AP)，公式为：AP = mean(rank / (rank + 1))，用于评估排序质量
            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    # 计算 mAP（均值平均精度）
    if valid_probe_sample_count > 0:
        mAP = avg_precision_sum / valid_probe_sample_count
        # print(f"mAP: {mAP * 100}")
    else:
        print("Warning: No valid probe samples found. mAP calculation cannot be performed.")
        mAP = 0  # 如果没有有效的查询样本，则返回 0
    return mAP

def get_mAP_for_teacher(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]

    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0  # 有效查询样本数量
    avg_precision_sum = 0  # 平均精度和

    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]  # 取出result中的第probe_index行，包含了对于样本probe_index，与gallery中样本距离的远近

        # 判断查询样本与图库中样本是否匹配
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)  # 统计匹配数量

        if true_match_count != 0:  # 如果图库中有匹配的样本
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            # 计算平均精度 (AP)，公式为：AP = mean(rank / (rank + 1))，用于评估排序质量
            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    # 计算 mAP（均值平均精度）
    if valid_probe_sample_count > 0:
        mAP = avg_precision_sum / valid_probe_sample_count
        print(f"mAP: {mAP * 100}")
    else:
        print("Warning: No valid probe samples found. mAP calculation cannot be performed.")
        mAP = 0  # 如果没有有效的查询样本，则返回 0
    return mAP

# 评估函数，计算 mAP 和 CMC
def eval_regdb(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, rerank=False):
    # 正常情况下应该进行特征的归一化，这里已注释掉
    # gallery_feats = F.normalize(gallery_feats, dim=1)
    # query_feats = F.normalize(query_feats, dim=1)

    # 计算查询样本与图库样本的距离矩阵
    if rerank:
        dist_mat = re_ranking(query_feats, gallery_feats, eval_type=False)  # 如果需要重排序，则使用重排序函数
    else:
        dist_mat = pairwise_distance(query_feats, gallery_feats)  # 计算成对距离


    # print(f"Shape of dist_mat: {dist_mat.shape}")
    # print(f"Contents of dist_mat: \n{dist_mat}")
    # 对距离矩阵进行排序，得到排序后的索引
    sorted_indices = np.argsort(dist_mat, axis=1)
    # print(f"Shape of sorted_indices: {sorted_indices.shape}")
    # print(f"Contents of sorted_indices: \n{sorted_indices}")
    # 计算 mAP 和 CMC
    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    # 获取不同排名下的精度
    r1 = cmc[0] * 100
    r5 = cmc[4] * 100
    r10 = cmc[9] * 100
    r20 = cmc[19] * 100

    # 输出最终结果
    mAP = mAP * 100
    perf = 'r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f} , mAP = {:.2f}'
    logging.info(perf.format(r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20

def eval_regdb_for_teacher(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, rerank=False):
    # 正常情况下应该进行特征的归一化，这里已注释掉
    # gallery_feats = F.normalize(gallery_feats, dim=1)
    # query_feats = F.normalize(query_feats, dim=1)

    # 计算查询样本与图库样本的距离矩阵
    dist_mat = pairwise_distance(query_feats, gallery_feats)  # 计算成对距离
    if torch.allclose(dist_mat, dist_mat.T, atol=1e-6):
        num_rows = dist_mat.size(0)
        diag_indices = torch.arange(num_rows, device=dist_mat.device)  # 获取0到num_rows-1的索引
        dist_mat[diag_indices, diag_indices] = 10000.0
    # print(f"Shape of dist_mat: {dist_mat.shape}")
    # print(f"Contents of dist_mat: \n{dist_mat}")
    # 对距离矩阵进行排序，得到排序后的索引
    sorted_indices = np.argsort(dist_mat, axis=1)
    # print(f"Shape of sorted_indices: {sorted_indices.shape}")
    # print(f"Contents of sorted_indices: \n{sorted_indices}")

    # 计算 mAP 和 CMC
    mAP = get_mAP_for_teacher(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc_for_teacher(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    # 获取不同排名下的精度
    r1 = cmc[0] * 100
    r5 = cmc[4] * 100
    r10 = cmc[9] * 100
    r20 = cmc[19] * 100

    # 输出最终结果
    mAP = mAP * 100
    perf = 'r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f} , mAP = {:.2f}'
    logging.info(perf.format(r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
