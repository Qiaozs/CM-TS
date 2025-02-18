import math
import torch
import torch.nn as nn
from utils.calc_acc import calc_acc
from layers import TripletLoss, RerankLoss
from torch.nn import functional as F


from models.IBN_ResNet import resnet50_ibn_a
from models.ResNet import resnet50

def gem(x, p=3, eps=1e-6):
    # x.clamp(min=eps):对输入的特征图进行裁剪，确保所有值不小于 eps，防止后续的 pow(p) 操作导致数值溢出
    # x.pow(p): 对裁剪后的特征图每个元素计算 p 次幂
    # F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))): 返回的结果是 C0 通道的通道池化结果
    # pow(1. / p): 对池化结果再计算 1/p 次幂。
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

def Bg_kl(logits1, logits2):####输入:(60,206),(60,206)
    '''
    双向 KL 散度（Kullback-Leibler Divergence）损失计算
    Args:
    Returns:
    返回一个单向KL损失，一个双向KL损失
    '''
    KL = nn.KLDivLoss(reduction='batchmean')
    # 默认要求第一个输入为对数概率分布（log_softmax），第二个输入为普通概率分布（softmax）。
    # 将每个样本的 KL 散度计算结果取平均（按批次）
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    bg_loss_kl = kl_loss_12 + kl_loss_21  # 相加作为双向 KL 损失。
    return kl_loss_12, bg_loss_kl

def compute_centroid_distance(features, labels, modalities):
    """
    计算每个类别不同模态的中心特征的距离。

    参数:
    features -- 特征矩阵，形状为(B, C)。
    labels -- 类别标签，形状为(B,)。
    modalities -- 模态标签，形状为(B,)。

    返回:
    distances -- 每个类别模态中心距离的列表。
    """
    unique_labels = torch.unique(labels)  # 从输入的标签张量 labels 中提取唯一的标签值
    distances = []
    for label in unique_labels:
        # 分别获取当前类别下的两种模态的特征
        features_modality_0 = features[(labels == label) & (modalities == 0)]
        features_modality_1 = features[(labels == label) & (modalities == 1)]

        # 计算中心特征
        centroid_modality_0 = features_modality_0.mean(dim=0)
        centroid_modality_1 = features_modality_1.mean(dim=0)

        # 计算两个中心特征之间的距离，这里使用欧氏距离
        distance = F.pairwise_distance(centroid_modality_0.unsqueeze(0), centroid_modality_1.unsqueeze(0))
        distances.append(distance)


    return torch.stack(distances)


def cosine_similarity_matrix(features, matrix):

    # 计算余弦相似度
    device = torch.device("cuda")  # 使用CUDA进行GPU加速

    normalized_features = F.normalize(features, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(normalized_features, normalized_features.T)
    matrix = matrix.to(device)
    cosine_sim_matrix = cosine_sim_matrix.to(device)
    mse_loss = F.mse_loss(cosine_sim_matrix, matrix)
    return mse_loss

class Mask(nn.Module):
    def __init__(self, dim, r=16):
        super(Mask, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            # 卷积层1：将输入dim降维至dim // r
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            # 卷积层2：将dim // r升维至dim
            nn.Sigmoid()
            # Sigmoid将输出限制在 [0, 1] 的范围，作为通道注意力的权重掩码（mask）。
        )

    def forward(self, x):
        mask = self.channel_attention(x)
        return mask


class RGB_Teacher(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(RGB_Teacher, self).__init__()
        self.backbone = resnet50_ibn_a(pretrained=True)
        self.IN = nn.InstanceNorm2d(2048, track_running_stats=True, affine=True)
        self.mask = Mask(2048)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x_IN = self.IN(x)
        m_IN = self.mask(x_IN)
        x = x_IN * m_IN
        x = gem(x).squeeze()  # Gem池化
        x = x.view(x.size(0), -1)  # Gem池化
        # x = self.backbone(x)

        x = self.classifier(x)

        return x

class IR_Teacher(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(IR_Teacher, self).__init__()

        self.IR_Teacher_resnet = resnet50_ibn_a(pretrained=True)
        self.IN = nn.InstanceNorm2d(2048, track_running_stats=True, affine=True)
        self.mask = Mask(2048)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        x = self.IR_Teacher_resnet.conv1(x)
        x = self.IR_Teacher_resnet.bn1(x)
        x = self.IR_Teacher_resnet.relu(x)
        x = self.IR_Teacher_resnet.maxpool(x)
        x = self.IR_Teacher_resnet.layer1(x)
        x = self.IR_Teacher_resnet.layer2(x)
        x = self.IR_Teacher_resnet.layer3(x)
        x = self.IR_Teacher_resnet.layer4(x)
        x_IN = self.IN(x)
        m_IN = self.mask(x_IN)
        x = x_IN * m_IN
        x = gem(x).squeeze()  # Gem池化
        x = x.view(x.size(0), -1)  # Gem池化
        # x = self.backbone(x)
        x = self.classifier(x)

        return x

class CrossModality_Student(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(CrossModality_Student, self).__init__()

        self.backbone = resnet50(pretrained=True)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = gem(x).squeeze()  # Gem池化
        x = x.view(x.size(0), -1)  # Gem池化
        x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(Net,self).__init__()
        self.mutual_learning = kwargs.get('mutual_learning',False)
        self.drop_last_stride = drop_last_stride
        self.Teaching = kwargs.get('Teaching', False)
        self.RGB_Teaching = kwargs.get('RGB_Teaching', False)
        self.IR_Teaching = kwargs.get('IR_Teaching', False)
        self.margin = kwargs.get('margin', 0.3)  # 用于三元组损失或中心聚类的边际值
        self.p_size = kwargs.get('p_size', 8)
        self.k_size = kwargs.get('k_size', 10)
        self.Batch_size = self.p_size * self.k_size

        self.RGB_Training = kwargs.get('RGB_Training', False)
        self.IR_Training = kwargs.get('IR_Training', False)
        self.Student_Training = kwargs.get('Student_Training', False)


        self.RGB_Teacher = RGB_Teacher(num_classes=num_classes)
        self.IR_Teacher = IR_Teacher(num_classes=num_classes)
        self.CrossModality_Student = CrossModality_Student(num_classes=num_classes)
        self.classifier = nn.Linear(1000, num_classes, bias=False)
        self.modalityClassifier = nn.Linear(1000, 2, bias=False)
        # self.modalityConfuser = nn.Linear(1000, 2, bias=False)

        # self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.id_loss = nn.CrossEntropyLoss()

        self.triplet_loss = TripletLoss(margin=self.margin)

    def forward(self, total_input, rgb_input, ir_input,
                total_labels=None, rgb_labels=None, ir_labels=None,
                rgb_test=False, ir_test=False,
                **kwargs):

        if self.training:
            feat_rgb = None
            feat_ir = None
            feat_student = None
            if self.RGB_Training:
                x_rgb = rgb_input
                feat_rgb = self.RGB_Teacher(x_rgb)
            if self.IR_Training:
                x_ir = ir_input
                feat_ir = self.IR_Teacher(x_ir)
            if self.Student_Training:
                x_student = total_input
                feat_student = self.CrossModality_Student(x_student)
            # print(feat_student.shape)

            # x_student = total_input
            # x_rgb = rgb_input
            # x_ir = ir_input
            # feat_student = self.CrossModality_Student(x_student)
            # feat_rgb = self.RGB_Teacher(x_rgb)
            # feat_ir = self.IR_Teacher(x_ir)

            return self.train_forward(feat_student, feat_rgb, feat_ir,

                                      total_labels, rgb_labels, ir_labels,
                                      **kwargs)

        else:
            x_student = total_input
            feat_student, logit_student = self.CrossModality_Student(x_student)
            if rgb_test:
                # print("rgb_test:",rgb_test)
                x_rgb_query = rgb_input
                x_rgb_gallery = ir_input
                feat_rgb_query, logit_rgb_query = self.RGB_Teacher(x_rgb_query)
                feat_rgb_gallery, logit_rgb_gallery = self.RGB_Teacher(x_rgb_gallery)
                return logit_student, logit_rgb_query, logit_rgb_gallery
            elif ir_test:
                # print("ir_test:",ir_test)
                x_ir_query = rgb_input
                x_ir_gallery = ir_input
                feat_ir_query, logit_ir_query = self.IR_Teacher(x_ir_query)
                feat_ir_gallery, logit_ir_gallery = self.IR_Teacher(x_ir_gallery)
                return logit_student, logit_ir_query, logit_ir_gallery
            # else:
            #     return feat_student

    def train_forward(self, feat_student=None, feat_rgb=None, feat_ir=None,
                      total_labels=None, rgb_labels=None, ir_labels=None, **kwargs):
        total_cam_ids = kwargs.get('total_cam_ids')
        sub = (total_cam_ids == 3) + (total_cam_ids == 6)  # sub 标记特征是否来自红外图像，sub为True即为红外图像

        device = torch.device("cuda")  # 使用CUDA进行GPU加速
        epoch = kwargs.get('epoch')
        metric = {}
        loss_student = 0  # 初始化loss
        loss_rgb = 0
        loss_ir = 0

        if self.RGB_Training:
            rgb_cls_loss = self.id_loss(feat_rgb.float(), rgb_labels)
            loss_rgb += rgb_cls_loss
            metric.update({'rgb_acc': calc_acc(feat_rgb.data, rgb_labels), 'rgb_cls_loss': rgb_cls_loss.data})

            rgb_triplet_loss, _, _, _ = self.triplet_loss(feat_rgb.float(), rgb_labels)  # 输入波浪线f_sp，特征标签
            loss_student += rgb_triplet_loss
            metric.update({'rgb_triplet_loss': rgb_triplet_loss.data})

        if self.IR_Training:
            ir_cls_loss = self.id_loss(feat_ir.float(), ir_labels)
            loss_ir += ir_cls_loss
            metric.update({'ir_acc': calc_acc(feat_ir.data, ir_labels), 'ir_cls_loss': ir_cls_loss.data})

            ir_triplet_loss, _, _, _ = self.triplet_loss(feat_ir.float(), ir_labels)  # 输入f_sh，特征标签
            loss_student += ir_triplet_loss
            metric.update({'ir_triplet_loss': ir_triplet_loss.data})

        if self.Student_Training:
            total_cls_loss = self.id_loss(feat_student.float(), total_labels)
            loss_student += total_cls_loss
            metric.update({'total_acc': calc_acc(feat_student.data, total_labels), 'total_cls_loss': total_cls_loss.data})

            total_triplet_loss, _, _, _ = self.triplet_loss(feat_student.float(), total_labels)  # 输入f_sh，特征标签
            loss_student += total_triplet_loss
            metric.update({'total_triplet_loss': total_triplet_loss.data})

            matrix = torch.zeros(self.Batch_size, self.Batch_size, dtype=torch.int)
            # 通过循环填充每一行的1
            for i in range(self.Batch_size):
                start_idx = (i * self.k_size)  # 当前行起始为1的列索引
                end_idx = (i + 1) * self.k_size  # 当前行结束为1的列索引
                matrix[i, start_idx:end_idx] = 1  # 将相应的部分赋值为1

            loss_sim = cosine_similarity_matrix(feat_student.data, matrix)
            loss_student += loss_sim
            metric.update({'loss_sim': loss_sim.data})
            

            if self.Teaching:
                _, loss_test = Bg_kl(feat_student[sub == 0], feat_student[sub == 1])
                loss_student += loss_test
                metric.update({'loss_test': loss_test.data})

                if self.RGB_Teaching:
                    _, rgb_teaching_loss = Bg_kl(feat_student[sub == 1], feat_rgb)
                    loss_student += 0.1*rgb_teaching_loss
                    metric.update({'rgb_teaching_loss': rgb_teaching_loss.data})
                if self.IR_Teaching:
                    _, ir_teaching_loss = Bg_kl(feat_student[sub == 0], feat_ir)
                    loss_student += 0.1*ir_teaching_loss
                    metric.update({'ir_teaching_loss': ir_teaching_loss.data})

        return loss_student, loss_rgb, loss_ir, metric





