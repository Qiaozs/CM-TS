import numpy as np
import torch

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)  # 将查询特征展开为二维矩阵
    y = y.view(n, -1)  # 将图库特征展开为二维矩阵
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # dist为一个 m×n 的矩阵，描述m个query特征和n个gallery特征之间的距离
    return dist

# def re_ranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, eval_type=True):
#     # The following naming, e.g. gallery_num, is different from outer scope.
#     # Don't care about it.
#     feats = torch.cat([q_feat, g_feat], 0)
#     dist = pairwise_distance(feats, feats)  # 计算所有样本（包括查询和库存图像）之间的成对距离 （m+n）×（m+n）
#     original_dist = dist.detach().cpu().numpy()
#
#     all_num = original_dist.shape[0]
#     original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))  # 对距离矩阵进行规范化，使得每列的最大值为 1。
#     V = np.zeros_like(original_dist).astype(np.float16)  # 创建一个与original_dist形状相同的全零矩阵
#
#     query_num = q_feat.size(0)  # query集的特征数量
#     all_num = original_dist.shape[0]  # query和gallery集的特征数量
#     if eval_type:
#         dist[:, query_num:] = dist.max()
#     # 将 dist 矩阵中从 query_num 列开始的部分（即与查询图像无关的部分）设置为 dist 矩阵中的最大值。
#
#     dist = dist.detach().cpu().numpy()
#     initial_rank = np.argsort(dist).astype(np.int32)
#
#     # print("start re-ranking")
#     for i in range(all_num):
#         # k-reciprocal neighbors
#         # 首先选出与该样本最相似的 k1 个邻居，然后检查这些邻居是否与当前样本形成相互邻居（即两者互为邻居）
#         forward_k_neigh_index = initial_rank[i, :k1 + 1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#         fi = np.where(backward_k_neigh_index == i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
#                                                 :int(np.around(k1 / 2)) + 1]
#             # import pdb
#             # pdb.set_trace()
#             fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
#                     candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#         weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
#         V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float16)
#         for i in range(all_num):
#             V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
#         V = V_qe
#         del V_qe
#     del initial_rank
#     invIndex = []
#     for i in range(all_num):
#         invIndex.append(np.where(V[:, i] != 0)[0])
#
#     jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
#
#
#     for i in range(query_num):
#         temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
#         indNonZero = np.where(V[i, :] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
#                                                                                 V[indImages[j], indNonZero[j]])
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num, query_num:]
#     # 返回最终的重排距离矩阵，只有查询样本和库存样本之间的距离。
#     return final_dist
#

def re_ranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, eval_type=True):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    feats = torch.cat([q_feat, g_feat], 0)

    # 计算所有样本（包括查询和图库图像）之间的成对距离 (m+n)×(m+n)
    dist = pairwise_distance(feats, feats)
    #
    # print("Pairwise distance matrix (dist) shape:", dist.shape)
    # print("Pairwise distance matrix (dist):", dist)

    original_dist = dist.detach().cpu().numpy()

    all_num = original_dist.shape[0]
    # print("Original distance matrix (original_dist) shape:", original_dist.shape)
    # print("Original distance matrix (original_dist):", original_dist)

    # 对距离矩阵进行规范化，使得每列的最大值为 1。
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))

    # print("Normalized original distance matrix (original_dist) shape:", original_dist.shape)
    # print("Normalized original distance matrix (original_dist):", original_dist)

    V = np.zeros_like(original_dist).astype(np.float16)  # 创建一个与original_dist形状相同的全零矩阵
    # print("V matrix initialized shape:", V.shape)
    # print("V matrix initialized:", V)

    query_num = q_feat.size(0)  # query集的特征数量
    all_num = original_dist.shape[0]  # query和gallery集的特征数量
    if eval_type:
        dist[:, query_num:] = dist.max()
        # print("Distance matrix after setting max value for gallery:", dist)

    dist = dist.detach().cpu().numpy()
    initial_rank = np.argsort(dist).astype(np.int32)
    # print("Initial rank indices (initial_rank) shape:", initial_rank.shape)
    # print("Initial rank indices (initial_rank):", initial_rank)

    # 开始重排序
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        # print(f"forward_k_neigh_index for sample {i}: {forward_k_neigh_index}")

        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        # print(f"backward_k_neigh_index for sample {i}: {backward_k_neigh_index}")

        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        # print(f"k_reciprocal_index for sample {i}: {k_reciprocal_index}")

        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            # print(f"candidate_backward_k_neigh_index for sample {candidate}: {candidate_backward_k_neigh_index}")

            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            # print(f"candidate_k_reciprocal_index for sample {candidate}: {candidate_k_reciprocal_index}")

            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        # print(f"k_reciprocal_expansion_index for sample {i}: {k_reciprocal_expansion_index}")

        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        # print(f"Weight for k-reciprocal expansion for sample {i}: {weight}")

        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, :]
    # print("Original distance matrix after trimming (original_dist):", original_dist)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
        # print("V after k2 expansion:", V)

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    # print("invIndex:", invIndex)

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    # print("Initial jaccard_dist matrix shape:", jaccard_dist.shape)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        # print(f"indNonZero for sample {i}: {indNonZero}")

        indImages = [invIndex[ind] for ind in indNonZero]
        # print(f"indImages for sample {i}: {indImages}")

        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        # print(f"jaccard_dist for sample {i}: {jaccard_dist[i]}")

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    # print("Final distance matrix (final_dist) shape:", final_dist.shape)
    # print("Final distance matrix (final_dist):", final_dist)

    # 删除不再需要的矩阵
    del original_dist
    del V
    del jaccard_dist

    # 仅返回查询和库存之间的距离矩阵
    final_dist = final_dist[:query_num, query_num:]
    # print("Final re-ranked distance matrix (final_dist) after trimming:", final_dist)

    return final_dist
