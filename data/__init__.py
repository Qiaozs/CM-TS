import os

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader

from .dataset import RegDBDataset
from.dataset import SYSUDataset

from .sampler import SamplerForRegDB

import random


def collate_fn(batch):
    total_samples = list(zip(*batch))  # 将batch中的样本分解
    total_batch =[torch.stack(x, 0) for i, x in enumerate(total_samples) if i != 3]
    total_batch.insert(3, total_samples[3])

    rgb_samples = []
    ir_samples = []

    for img, label, cam_id, path, item in batch:
        if cam_id == 1 or cam_id == 2 or cam_id == 4 or cam_id == 5:  # RGB 模态
            rgb_samples.append((img, label, cam_id, path, item))
        elif cam_id == 3 or cam_id == 6:  # IR 模态
            ir_samples.append((img, label, cam_id, path, item))

    # 将RGB和IR样本堆叠成新的batch
    rgb_samples = list(zip(*rgb_samples)) if rgb_samples else ([], [], [], [], [])
    ir_samples = list(zip(*ir_samples)) if ir_samples else ([], [], [], [], [])

    rgb_batch = [torch.stack(x, 0) if len(x) > 0 else torch.empty(0) for i, x in enumerate(rgb_samples) if i != 3]
    ir_batch = [torch.stack(x, 0) if len(x) > 0 else torch.empty(0) for i, x in enumerate(ir_samples) if i != 3]
    rgb_batch.insert(3, rgb_samples[3])
    ir_batch.insert(3, ir_samples[3])

    # 输出调试信息，确保total_batch的长度为5
    # print(f"total_batch length: {len(total_batch)}")  # 调试：输出total_batch的长度
    assert len(total_batch) == 5, f"Expected 5 items in total_batch, but got {len(total_batch)}"

    # 返回 RGB 和 IR 数据的 batch
    return total_batch, rgb_batch, ir_batch

def test_collate_fn(batch):
    total_samples = list(zip(*batch))  # 将batch中的样本分解
    total_batch =[torch.stack(x, 0) for i, x in enumerate(total_samples) if i != 3]
    total_batch.insert(3, total_samples[3])
    assert len(total_batch) == 5, f"Expected 5 items in total_batch, but got {len(total_batch)}"

    for img, label, cam_id, path, item in batch:
        if cam_id == 1 or cam_id == 2 or cam_id == 4 or cam_id == 5:  # RGB 模态
            is_rgb = True
            break
        elif cam_id == 3 or cam_id == 6:  # IR 模态
            is_rgb = False
            break
    if is_rgb:
        rgb_samples_query = []
        rgb_samples_gallery = []
        for img, label, cam_id, path, item in batch:
            found_in_query = any(sample[1] == label for sample in rgb_samples_query)
            found_in_gallery = any(sample[1] == label for sample in rgb_samples_gallery)
            # 将同一个 ID 的图片均分成 query 和 gallery
            if not found_in_query and not found_in_gallery:
                rgb_samples_query.append((img, label, cam_id, path, item))  # 先添加到 query
            else:  # 在这里均分：前一半为 query，后一半为 gallery
                if len(rgb_samples_query) > len(rgb_samples_gallery):
                    rgb_samples_gallery.append((img, label, cam_id, path, item))
                else:
                    rgb_samples_query.append((img, label, cam_id, path, item))
        rgb_samples_query = list(zip(*rgb_samples_query)) if rgb_samples_query else ([], [], [], [], [])
        rgb_samples_gallery = list(zip(*rgb_samples_gallery)) if rgb_samples_gallery else ([], [], [], [], [])
        rgb_batch_query = [torch.stack(x, 0) if len(x) > 0 else torch.empty(0) for i, x in enumerate(rgb_samples_query) if i != 3]
        rgb_batch_gallery = [torch.stack(x, 0) if len(x) > 0 else torch.empty(0) for i, x in enumerate(rgb_samples_gallery) if i != 3]
        rgb_batch_query.insert(3, rgb_samples_query[3])
        rgb_batch_gallery.insert(3, rgb_samples_gallery[3])
        return total_batch, rgb_batch_query, rgb_batch_gallery
    else:
        ir_samples_query = []
        ir_samples_gallery = []
        for img, label, cam_id, path, item in batch:
            found_in_query = any(sample[1] == label for sample in ir_samples_query)
            found_in_gallery = any(sample[1] == label for sample in ir_samples_gallery)
            if not found_in_query and not found_in_gallery:
                ir_samples_query.append((img, label, cam_id, path, item))  # 先添加到 query
            else:  # 在这里均分：前一半为 query，后一半为 gallery
                if len(ir_samples_query) > len(ir_samples_gallery):
                    ir_samples_gallery.append((img, label, cam_id, path, item))
                else:
                    ir_samples_query.append((img, label, cam_id, path, item))
        ir_samples_query = list(zip(*ir_samples_query)) if ir_samples_query else ([], [], [], [], [])
        ir_samples_gallery = list(zip(*ir_samples_gallery)) if ir_samples_gallery else ([], [], [], [], [])
        ir_batch_query = [torch.stack(x, 0) if len(x) > 0 else torch.empty(0) for i, x in enumerate(ir_samples_query) if i != 3]
        ir_batch_gallery = [torch.stack(x, 0) if len(x) > 0 else torch.empty(0) for i, x in enumerate(ir_samples_gallery) if i != 3]
        ir_batch_gallery.insert(3, ir_samples_gallery[3])
        ir_batch_query.insert(3, ir_samples_query[3])
        # print("total_batch:", len(total_batch))
        # print("ir_batch_query:", len(ir_batch_query))
        # print("ir_batch_gallery:", len(ir_batch_gallery))
        return total_batch, ir_batch_query, ir_batch_gallery



class ChannelAdapGray(object):
    # 实现一种数据增强策略，对图像进行通道选择或灰度处理
    """ 自适应选择一个或两个通道。
    参数:
         probability: 执行随机擦除操作的概率。
         sl: 相对于输入图像的最小擦除区域比例。
         sh: 擦除区域与输入图像的最大比例。
         r1: 擦除区域的最小纵横比。
         mean: 擦除值。
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        # 通过 __call__ 方法对输入的图像进行通道选择或灰度转换操作

        idx = random.randint(0, 3)
        # idx为0-3的随机数

        if idx == 0:
            # random select R Channel
            # 图像的三个通道全部替换为R通道
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            # 图像的三个通道全部替换为B通道
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            # 图像的三个通道全部替换为G通道
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            # idx == 3
            if random.uniform(0, 1) > self.probability:
                # 产生0-1的随机数，如果大于probability，图像保持不变，如果小于probability，图像转换为灰度图
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img

def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=0):

    if True==False: #tsne 始终为false，不会被执行
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # data pre-processing
        t = [T.Resize(image_size)]
        # 调整图像大小
        if random_flip:
            t.append(T.RandomHorizontalFlip())
            # 随机水平翻转
        if color_jitter:
            t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
            # 随机调整图像的亮度、对比度、饱和度和色调。
        if random_crop:
            t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])
            # 随机裁剪
        t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # 转换为张量并归一化
        if random_erase:
            t.append(T.RandomErasing())
            # 随机擦除
            #t.append(ChannelAdapGray(probability=0.5)) ###58
            # t.append(Jigsaw())
        transform = T.Compose(t)

    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
        print("数据集SYSU，train模式下图片数量：", train_dataset.__len__())
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform)
        print("数据集RegDB，train模式下图片数量：", train_dataset.__len__())
    elif dataset == 'llcm':
        train_dataset = LLCMData(root, mode='train', transform=transform)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform)

    sampler = SamplerForRegDB(train_dataset, p_size, k_size)
    # loader
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              sampler=sampler,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    '''
        # DataLoader为torch.utils.data模块的类
        drop_last=True：控制是否丢弃数据集最后一个批次中的剩余样本。 
        pin_memory=True：针对 CUDA（GPU）加速的一个选项。 True：将数据加载到固定内存（pinned memory）中，以加速数据传输到 GPU。
        collate_fn=collate_fn：定义了如何将一个批次的样本组合成一个 mini-batch。
        num_workers=num_workers：定义了加载数据时使用的子进程数量。 
    '''
    return train_loader

def get_test_loader(dataset, root, batch_size, image_size, num_workers=0):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
        print("加载数据集SYSU测试数据，gallery/rgb_test模式下图片数量：", gallery_dataset.__len__(), ", query/ir_test模式下图片数量：",
              query_dataset.__len__())
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform)
        query_dataset = RegDBDataset(root, mode='query', transform=transform)
        print("加载数据集RegDB测试数据，gallery模式下图片数量：", gallery_dataset.__len__(), ", query模式下图片数量：",
              query_dataset.__len__())
    elif dataset == 'llcm':
        gallery_dataset = LLCMData(root, mode='gallery', transform=transform)
        query_dataset = LLCMData(root, mode='query', transform=transform)
        print("加载数据集LLCM测试数据，gallery模式下图片数量：", gallery_dataset.__len__(), ", query模式下图片数量：",
              query_dataset.__len__())
    elif dataset == 'market':
        gallery_dataset = MarketDataset(root, mode='gallery', transform=transform)
        query_dataset = MarketDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=test_collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=test_collate_fn,
                                num_workers=num_workers)


    # # 正确解包
    # for i, (total_batch, rgb_batch, ir_batch) in enumerate(gallery_loader):
    #     # 处理每个batch
    #     print(f"Batch {i}:")
    #     print(f"Total Batch Size: {len(total_batch[0])}")  # 访问总样本的第一个部分（图像）
    #     print(f"RGB Batch Size: {len(rgb_batch[0])}")  # 访问RGB样本
    #     print(f"IR Batch Size: {len(ir_batch[0])}")  # 访问IR样本
    #
    # for i, (total_batch, rgb_batch, ir_batch) in enumerate(query_loader):
    #     # 处理每个batch
    #     print(f"Batch {i}:")
    #     print(f"Total Batch Size: {len(total_batch[0])}")  # 访问总样本的第一个部分（图像）
    #     print(f"RGB Batch Size: {len(rgb_batch[0])}")  # 访问RGB样本
    #     print(f"IR Batch Size: {len(ir_batch[0])}")  # 访问IR样本

    return gallery_loader, query_loader
