import os
import logging
import torch
import numpy as np
# from sklearn.preprocessing import normalize
from .rerank import re_ranking, pairwise_distance
from torch.nn import functional as F


def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    '''

    :param perm: 含不同相机的排列信息。
    :param cams: 为[1,2]或者[1, 2, 4, 5]
    :param ids: 身份对象ID
    :param trial_id:
    :param num_shots:
    :return:
    '''
    names = []  # 存储生成的name
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        # [cam - 1] 数组索引从0开始
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])
    # names存储的内容形如cam3/0004/0002
    return names


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc
def get_cmc_for_teacher(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP
def get_mAP_for_teacher(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]


        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

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


def eval_sysu(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=1, num_trials=10, rerank=False):
    '''
    :param query_feats: query数据集的特征
    :param query_ids: query数据集的身份编号
    :param query_cam_ids: query数据集的相机编号
    :param gallery_feats: gallery数据集的特征
    :param gallery_ids: gallery数据集的身份编号
    :param gallery_cam_ids: gallery数据集的相机编号
    :param gallery_img_paths: gallery数据集的图像路径
    :param perm:
    :param mode:
    :param num_shots:
    :param num_trials:
    :param rerank:
    :return:
    '''
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]
    # indoor模式下使用[1,2]相机作为gallery集
    # all模式下使用[1,2,4,5]相机作为gallery集

    # cam2 and cam3 are in the same location
    # 把用户输入的query_cam_ids中为3的部分都改为2
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2

    # 特征归一化
    query_feats = F.normalize(query_feats, dim=1)

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)  # 检查 gallery_cam_ids 数组中的元素是否存在于 gallery_cams
    # gallery_indices为一布尔数组，其代表了用户输入的gallery_cam_ids中的编号是否正确
    # eg：用户输入 gallery_cam_ids 为 [1, 2, 3, 4]，实际 gallery_cams 为 [1, 2]
    # 则 gallery_indices 为 [True, True, False, False]。
    
    gallery_feats = gallery_feats[gallery_indices]  # 提取来自正确cam的gallery特征
    gallery_feats = F.normalize(gallery_feats, dim=1)
    gallery_cam_ids = gallery_cam_ids[gallery_indices]  # 根据gallery_indices来将输入的gallery_cam_ids规范（去掉非gallery的cam_id）
    gallery_ids = gallery_ids[gallery_indices]  # 根据gallery_indices来将输入的gallery_ids规范（去掉非gallery的id）
    gallery_img_paths = gallery_img_paths[gallery_indices]  # 根据gallery_indices来将输入的gallery_img_paths规范（去掉非gallery的path）
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])
    # 借助分隔符从路径中提取最后三个作为gallery_names，如 SYSU-MM01/cam3/0004/0002.jpg -> cam3/0004/0002
    # gallery_names为 ：array(['images/2023/pic1', 'images/2023/pic2', 'documents/2023/report'])

    gallery_id_set = np.unique(gallery_ids)  # 提取gallery中的人物ID

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        # (perm, [1, 2], [001, 004, ...], t, 1)
        flag = np.in1d(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]

        if rerank:
            dist_mat = re_ranking(query_feats, g_feat)
        else:
            dist_mat = pairwise_distance(query_feats, g_feat)
            # dist_mat = -torch.mm(query_feats, g_feat.permute(1,0))

        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = '{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(mode, num_shots, r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20


def eval_sysu_for_teacher(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', mode1 ='rgb', num_shots=1, num_trials=10, rerank=False):
    '''
    :param query_feats: query数据集的特征
    :param query_ids: query数据集的身份编号
    :param query_cam_ids: query数据集的相机编号
    :param gallery_feats: gallery数据集的特征
    :param gallery_ids: gallery数据集的身份编号
    :param gallery_cam_ids: gallery数据集的相机编号
    :param gallery_img_paths: gallery数据集的图像路径
    :param perm:
    :param mode:
    :param num_shots:
    :param num_trials:
    :param rerank:
    :return:
    '''
    assert mode in ['indoor', 'all']
    assert mode1 in ['rgb', 'ir']
    if mode == 'all' and mode1 == 'rgb':
        gallery_cams = [1, 2, 4, 5]
    elif mode == 'indoor' and mode1 == 'rgb':
        gallery_cams = [1, 2]
    elif mode == 'all' and mode1 == 'ir':
        gallery_cams = [3, 6]
    else:
        gallery_cams = [3, 6]

    gallery_feats = F.normalize(gallery_feats, dim=1)
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])
    # 借助分隔符从路径中提取最后三个作为gallery_names，如 SYSU-MM01/cam3/0004/0002.jpg -> cam3/0004/0002
    # gallery_names为 ：array(['images/2023/pic1', 'images/2023/pic2', 'documents/2023/report'])

    gallery_id_set = np.unique(gallery_ids)  # 提取gallery中的人物ID

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        # (perm, [1, 2], [001, 004, ...], t, 1)
        flag = np.in1d(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]
        # print('g_feat', g_feat.shape)
        # print('query_feats',query_feats.shape)
        if rerank:
            dist_mat = re_ranking(query_feats, g_feat)
        else:
            dist_mat = pairwise_distance(query_feats, g_feat)
            # dist_mat = -torch.mm(query_feats, g_feat.permute(1,0))

        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP_for_teacher(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc_for_teacher(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = '{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(mode, num_shots, r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
