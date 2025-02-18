import numpy as np

import copy
from torch.utils.data import Sampler
from collections import defaultdict

class SamplerForRegDB(Sampler):
    def __init__(self, regdb_dataset, k_size, p_size):
        self.regdb_dataset = regdb_dataset
        self.batch_size = p_size * k_size
        self.num_instances = p_size
        self.num_pids_per_batch = self.batch_size // self.num_instances  # 一个batch中的身份个数
        self.length = 0
        self.index_dic_R = defaultdict(list)
        self.index_dic_I = defaultdict(list)

        for i, identity in enumerate(regdb_dataset.ids):
            # index_dic_I保存每个身份在IR模态下的索引
            # index_dic_R 保存每个身份在RGB模态下的索引
            if regdb_dataset.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_R[identity].append(i)
        self.pids = list(self.index_dic_I.keys())
        # print("pids: ", self.pids)
        # self.pids：IR 模态中所有身份的列表。
        # estimate number of examples in an epoch
        self.length = 0  # 估计一个 epoch 中的总样本数量。
        for pid in self.pids:  # 遍历所有身份（person IDs）
            idxs = self.index_dic_I[pid]
            num = len(idxs)  # 计算当前身份pid在IR模态内的图片个数
            if num < self.num_instances:
                num = self.num_instances   # 如果当前身份pid在IR模态内的图片个数不够，则需要补足（通过有放回采样来补足）。
            self.length += num - num % self.num_instances  # 累加有效的样本数量

    def __iter__(self):
        # 用于构建训练批次
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:  # 遍历每个身份
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_R = copy.deepcopy(self.index_dic_R[pid])

            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                # 若该身份下IR模态内的样本个数<num_instances的一半，且该身份下RGB模态内的样本个数也<num_instances的一半
                # 即该身份的两个模态样本都需要补足，所以进行如下有放回采样
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_R):
                # 如果该身份的IR样本多于RGB样本，则对IR随机采样使其大小与RGB相同
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)
            if len(idxs_R) > len(idxs_I):
                # 同理
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)
            # 随机打乱样本
            batch_idxs_R = []
            batch_idxs_I = []

            for idx_I, idx_R in zip(idxs_I, idxs_R):
                # 每次从 IR 和 RGB 索引中取出一对样本（idx_I 和 idx_R），交替添加到 batch_idxs。
                batch_idxs_R.append(idx_R)
                batch_idxs_I.append(idx_I)
                # print("batch_idxs_R:", batch_idxs_R)
                # print("batch_idxs_I:", batch_idxs_I)
                if len(batch_idxs_R) == self.num_instances // 2:
                    # 如果当前身份的样本个数达到了 num_instances，将其存入 batch_idxs_dict[pid]，并清空 batch_idxs 以构造新的批次
                    batch_idxs_dict[pid].append(batch_idxs_R+batch_idxs_I)
                    # print("batch_idxs_dict[", pid, "] =", batch_idxs_dict[pid])
                    batch_idxs_R = []
                    batch_idxs_I = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        # 构造最终批次

        while len(avai_pids) >= self.num_pids_per_batch:  # 循环添加直到批次身份达到num_pids_per_batch
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            # 从avai_pids身份列表终随机选择num_pids_per_batch个身份作为selected_pids
            for pid in selected_pids:
                # 对num_pids_per_batch个身份依次执行
                batch_idxs = batch_idxs_dict[pid].pop(0)
                # 从经过上述循环的batch_idxs_dict[pid]字典中提取第一个批次pop(0)
                final_idxs.extend(batch_idxs)
                # print("final_idxs_dict={}".format(final_idxs))
                # print("pids为:", pid, "的身份的图片索引为：" , batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    # 如果当前身份的样本已经用尽，则将其从avai_pids中剔除
                    avai_pids.remove(pid)

        # while len(avai_pids) >= self.num_pids_per_batch:
        #     selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
        #     print(f"Remaining pids: {len(avai_pids)}")  # 打印剩余身份数
        #     for pid in selected_pids:
        #         # 确保从 batch_idxs_dict 中取出并更新 avai_pids
        #         if batch_idxs_dict[pid]:
        #             batch_idxs = batch_idxs_dict[pid].pop(0)
        #             final_idxs.extend(batch_idxs)
        #             print(f"Adding batch for pid {pid}: {batch_idxs}")
        #             if not batch_idxs_dict[pid]:
        #                 avai_pids.remove(pid)
        #                 print(f"Removed pid {pid} from remaining pids")
        #
        #     print(f"Remaining pids after update: {len(avai_pids)}")  # 更新后的剩余身份数

        self.length = len(final_idxs)
        # print("final_idxs: ", final_idxs)
        # 计算最终批次的大小
        return iter(final_idxs)

    def __len__(self):
        return self.length