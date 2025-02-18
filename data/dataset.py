import os
import re
import os.path as osp
from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
    提取各个dataset的图片信息，返回：
    img：经过预处理的图片（ PyTorch 张量）。
    label：图片对应的行人 ID（PyTorch 张量）。 
    cam：图片对应的摄像头 ID（PyTorch 张量）。
    path：图片的文件路径（字符串）。
    item：当前样本的索引（PyTorch 张量）。
'''


class SYSUDataset(Dataset):

    def __init__(self, root, mode='train', transform=None):

        # 更改root指定数据集路径
        assert os.path.isdir(root)
        # 检测root是否有效
        assert mode in ['train', 'gallery', 'query', 'rgb_test', 'ir_test']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()
            # 读取 train_id.txt 和 val_id.txt 文件的第一行内容，分别包含训练集和验证集的人物ID。
            # 文件路径为/root/exp/train_id.txt

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            # 将字符串转换为列表
            # strip('\n')去除字符串开头和末尾的换行符，确保读取到的行人 ID 字符串没有换行符
            # split(',') 将字符串按逗号分隔，得到列表

            selected_ids = train_ids + val_ids
            # 拼接为一个新的列表
        elif mode == 'gallery' or mode == 'rgb_test':
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        # 将字符串类型的行人ID转换为整数
        num_ids = len(selected_ids)
        # 计算ID的数量

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        # glob函数查找符合指定模式的文件路径
        # 此处查找root路径下的所有jpg格式文件，recursive=True表示允许递归搜索
        # img_paths是包含所有找到的 .jpg 文件的完整路径列表。
        # 示例路径：/root/SYSUDataset/cam1/0001/0002.jpg
        img_paths = [os.path.normpath(path) for path in img_paths]  # 规范化路径分隔符
        # print("Image paths before filtering:", img_paths)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]
        # img_paths = [path for path in img_paths if int(os.path.basename(os.path.dirname(path))) in selected_ids]
        # path.split('/')[-2]：将路径按 / 分割，取倒数第二个部分，这个部分通常表示行人 ID
        # img_paths保存了满足条件的路径【在selected_ids中的，来自训练验证集（mode为train）or测试集（mode为其他）】

        if mode == 'gallery' or mode == 'rgb_test':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
            # 在 gallery模式下，只选取来自cam1,2,4,5的图片路径（可见光图片）
        elif mode == 'query' or mode == 'ir_test':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]
            # 在 query模式下，只选取来自cam3,6的图片路径（红外光图片）

        img_paths = sorted(img_paths)
        # 将路径按字母排序
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids  # 行人ID总数
        self.transform = transform
        # 保存img_paths、cam_ids、num_ids等

        if mode == 'train':
            # 若mode为train，selected_ids为训练集和验证集的行人ID
            id_map = dict(zip(selected_ids, range(num_ids)))
            # zip(selected_ids, range(num_ids))：将 selected_ids 中的实际行人 ID 和 range(num_ids) 中的连续数字配对。
            # dic将配对映射为字典
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
            # self.ids 保存每张图片对应的行人 ID 的映射索引。
        else:
            # 若mode为其他，selected_ids为测试集的行人ID
            self.ids = [int(path.split('/')[-2]) for path in img_paths]
            # self.ids 保存每张图片对应的行人 ID，而不是映射索引

            # 训练模式下将实际的行人 ID 映射为从 0 开始的连续索引，模型的输出维度可以与行人类别数一致（如 num_ids）。
            # 非训练模式不需要映射，因为在测试阶段，行人ID是实际标签，用于评估模型的性能。

    def __len__(self):
        # print("图片个数：", len(self.img_paths))
        return len(self.img_paths)

    def __getitem__(self, item):
        '''
        用于创建batch，将输出的img, label, cam, path, item封装到一个batch中
        :param item:
        :return:
        '''
        path = self.img_paths[item]  # 从img_paths中获取索引为item的图片路径
        img = Image.open(path)  # 打开图片
        if self.transform is not None:
            # 若 transform不为None，对图片进行数据增强和转换
            img = self.transform(img)

        # if item == 5 :
        #    print("SYSU-MM01数据集加载示例：img6:", self.img_paths[item],
        #       ",行人ID:", self.ids[item],
        #       ",cam:", self.cam_ids[item],
        #       ",索引item:", item)

        label = torch.tensor(self.ids[item], dtype=torch.long)  # 提取标签信息
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)  # 提取相机号
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_' + num + '.txt', 'r'))
            index_IR = loadIdx(open(root + '/idx/train_thermal_' + num + '.txt', 'r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_' + num + '.txt', 'r'))
            index_IR = loadIdx(open(root + '/idx/test_thermal_' + num + '.txt', 'r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths]
        # the visible cams are 1 2 4 5 and thermal cams are 3 6 in sysu
        # to simplify the code, visible cam is 2 and thermal cam is 3 in regdb
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


class LLCMData(Dataset):
    def __init__(self, root, mode='train', transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_vis.txt', 'r'))
            index_IR = loadIdx(open(root + '/idx/train_nir.txt', 'r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_vis.txt', 'r'))
            index_IR = loadIdx(open(root + '/idx/test_nir.txt', 'r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)
        # path = '/home/zhang/E/RKJ/MAPnet/dataset/LLCM/nir/0351/0351_c06_s200656_f4830_nir.jpg'
        # img = Image.open(path).convert('RGB')
        # img = np.array(img, dtype=np.uint8)
        # import pdb
        # pdb.set_trace()

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'nir') + 2 for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))

            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


class MarketDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        self.transform = transform

        if mode == 'train':
            img_paths = glob(os.path.join(root, 'bounding_box_train/*.jpg'), recursive=True)
        elif mode == 'gallery':
            img_paths = glob(os.path.join(root, 'bounding_box_test/*.jpg'), recursive=True)
        elif mode == 'query':
            img_paths = glob(os.path.join(root, 'query/*.jpg'), recursive=True)

        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        relabel = mode == 'train'
        self.img_paths = []
        self.cam_ids = []
        self.ids = []
        for fpath in img_paths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            self.img_paths.append(fpath)
            self.ids.append(all_pids[pid])
            self.cam_ids.append(cam - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
