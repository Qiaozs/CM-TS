import math
import torch
import torch.nn as nn
from utils.calc_acc import calc_acc
# from layers import TripletLoss, RerankLoss
from layers.loss.triplet import TripletLoss

from torch.nn import functional as F

from models.IBN_ResNet import resnet50_ibn_a
from models.ResNet import resnet50


def gem(x, p=3, eps=1e-6):
    # x.clamp(min=eps):对输入的特征图进行裁剪，确保所有值不小于 eps，防止后续的 pow(p) 操作导致数值溢出
    # x.pow(p): 对裁剪后的特征图每个元素计算 p 次幂
    # F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))): 返回的结果是 C0 通道的通道池化结果
    # pow(1. / p): 对池化结果再计算 1/p 次幂。
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


def Bg_kl(logits1, logits2):  ####输入:(60,206),(60,206)
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


def pairwise_dist(x, y):
    # 计算两个矩阵之间的两两向量的距离（欧几里得距离）
    xx = (x**2).sum(dim=1, keepdim=True)
    # 计算 x 中每个向量的平方和，结果是一个形状为 (n, 1) 的列向量。
    yy = (y**2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2.0 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist


def kl_soft_dist(feat1,feat2):
    '''
    计算两个特征集合之间的两两距离
    '''
    n_st = feat1.size(0)  # n_st为特征矩阵1的特征维度
    dist_st = pairwise_dist(feat1, feat2)  # 调用pairwise_dist（）计算矩阵中向量距离
    mask_st_1 = torch.ones(n_st, n_st, dtype=bool)  # 全True方阵
    for i in range(n_st):  # 将同一类样本中自己与自己的距离舍弃
        mask_st_1[i, i] = 0
    dist_st_2 = []
    for i in range(n_st):
        dist_st_2.append(dist_st[i][mask_st_1[i]])
    dist_st_2 = torch.stack(dist_st_2)
    return dist_st_2

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
        # [B, 3, 256, 128]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # [B, 2048, 8, 4]
        x_IN = self.IN(x)
        m_IN = self.mask(x_IN)
        x = x_IN * m_IN
        x = gem(x).squeeze()  # Gem池化
        feat = x.view(x.size(0), -1)  # Gem池化
        x = self.classifier(feat)
        return feat, x


class IR_Teacher(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(IR_Teacher, self).__init__()

        self.backbone = resnet50_ibn_a(pretrained=True)
        self.IN = nn.InstanceNorm2d(2048, track_running_stats=True, affine=True)
        self.mask = Mask(2048)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        # [B, 3, 256, 128]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # [B, 2048, 8, 4]
        x_IN = self.IN(x)
        m_IN = self.mask(x_IN)
        x = x_IN * m_IN
        x = gem(x).squeeze()  # Gem池化
        feat = x.view(x.size(0), -1)  # Gem池化
        x = self.classifier(feat)
        return feat, x


class CrossModality_Student(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(CrossModality_Student, self).__init__()

        self.backbone_resnet50 = resnet50(pretrained=True)
        self.backbone_ibn_resnet50 = resnet50_ibn_a(pretrained=True)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        x = self.backbone_resnet50(x)
        # [B,2048,12,5]
        x = gem(x).squeeze()  # Gem池化
        feat = x.view(x.size(0), -1)  # Gem池化
        x = self.classifier(feat)
        return feat, x

class Net(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(Net, self).__init__()
        self.mutual_learning = kwargs.get('mutual_learning', False)
        self.drop_last_stride = drop_last_stride
        self.Teaching = kwargs.get('Teaching', False)
        self.RGB_Teaching = kwargs.get('RGB_Teaching', False)
        self.IR_Teaching = kwargs.get('IR_Teaching', False)
        self.margin = kwargs.get('margin', 0.3)
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
            logit_student = None
            logit_rgb = None
            logit_ir = None
            if self.RGB_Training:
                x_rgb = rgb_input
                feat_rgb, logit_rgb = self.RGB_Teacher(x_rgb)
            if self.IR_Training:
                x_ir = ir_input
                feat_ir, logit_ir = self.IR_Teacher(x_ir)
            if self.Student_Training:
                x_student = total_input
                feat_student, logit_student = self.CrossModality_Student(x_student)

            return self.train_forward(feat_student, feat_rgb, feat_ir,
                                      logit_student, logit_rgb, logit_ir,
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
                return logit_student, feat_student, feat_rgb_query, feat_rgb_gallery
            elif ir_test:
                # print("ir_test:",ir_test)
                x_ir_query = rgb_input
                x_ir_gallery = ir_input
                feat_ir_query, logit_ir_query = self.IR_Teacher(x_ir_query)
                feat_ir_gallery, logit_ir_gallery = self.IR_Teacher(x_ir_gallery)
                return logit_student, feat_student, feat_ir_query, feat_ir_gallery

    def train_forward(self, feat_student=None, feat_rgb=None, feat_ir=None,
                      logit_student=None, logit_rgb=None, logit_ir=None,
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
            rgb_cls_loss = self.id_loss(logit_rgb.float(), rgb_labels)
            loss_rgb += rgb_cls_loss
            metric.update({'rgb_acc': calc_acc(logit_rgb.data, rgb_labels), 'rgb_cls_loss': rgb_cls_loss.data})

            # rgb_triplet_loss, _ = self.triplet_loss(logit_rgb.float(), rgb_labels)
            # loss_student += rgb_triplet_loss
            # metric.update({'rgb_triplet_loss': rgb_triplet_loss.data})

        if self.IR_Training:
            ir_cls_loss = self.id_loss(logit_ir.float(), ir_labels)
            loss_ir += ir_cls_loss
            metric.update({'ir_acc': calc_acc(logit_ir.data, ir_labels), 'ir_cls_loss': ir_cls_loss.data})

            # ir_triplet_loss, _ = self.triplet_loss(logit_ir.float(), ir_labels)
            # loss_student += ir_triplet_loss
            # metric.update({'ir_triplet_loss': ir_triplet_loss.data})

        if self.Student_Training:
            total_cls_loss = self.id_loss(logit_student.float(), total_labels)
            loss_student += total_cls_loss
            metric.update({'total_acc': calc_acc(logit_student.data, total_labels), 'total_cls_loss': total_cls_loss.data})

            total_triplet_loss, _ = self.triplet_loss(logit_student.float(), total_labels)
            loss_student += total_triplet_loss
            metric.update({'total_triplet_loss': total_triplet_loss.data})

            matrix = torch.zeros(self.Batch_size, self.Batch_size, dtype=torch.int)
            # 通过循环填充每一行的1
            for i in range(self.Batch_size):
                start_idx = (i * self.k_size)
                end_idx = (i + 1) * self.k_size
                matrix[i, start_idx:end_idx] = 1

            # normalized_features = F.normalize(feat_student.data, p=2, dim=1)
            # cosine_sim_matrix = torch.matmul(normalized_features, normalized_features.T)
            # # print(cosine_sim_matrix)
            # loss_sim_feat_softmax = 0
            # SN = torch.zeros(self.Batch_size, device=device)
            # for i in range(self.Batch_size):
            #     for j in range(self.Batch_size):
            #         if total_labels[i] != total_labels[j]:
            #             SN[i] += torch.exp(cosine_sim_matrix[i][j])
            # for i in range(self.Batch_size):
            #     softmax_loss_i = 0
            #     for j in range(self.Batch_size):
            #         if total_labels[i] == total_labels[j] and i!=j:
            #             frac_up = torch.exp(cosine_sim_matrix[i][j])
            #             frac_down = frac_up + SN[i]
            #             softmax_loss_i += -torch.log(frac_up / frac_down)
            #     loss_sim_feat_softmax += softmax_loss_i
            # # print(type(loss_sim_feat_softmax))  # 确认它是否是 tensor 类型
            #
            # loss_student += loss_sim_feat_softmax
            # metric.update({'loss_sim_feat_softmax': loss_sim_feat_softmax})

            # loss_sim_feat = cosine_similarity_matrix(feat_student.data, matrix)
            # loss_student += loss_sim_feat
            # metric.update({'loss_sim_feat': loss_sim_feat.data})

            loss_sim_logit = cosine_similarity_matrix(logit_student.data, matrix)
            loss_student += loss_sim_logit
            metric.update({'loss_sim_logit': loss_sim_logit.data})

            if self.Teaching:
                distance_rgb_student = kl_soft_dist(feat_student[sub == 0], feat_student[sub == 0])
                distance_ir_student = kl_soft_dist(feat_student[sub == 1], feat_student[sub == 1])
                _, loss_student_cross_modality = Bg_kl(distance_rgb_student, distance_ir_student)
                loss_student += loss_student_cross_modality
                metric.update({'loss_student_cross_modality': loss_student_cross_modality.data})

                distance_rgb_ir_student = kl_soft_dist(feat_student[sub == 1], feat_student[sub == 0])
                # print("distance_rgb_ir_student: ", distance_rgb_ir_student)

                if self.RGB_Teaching:
                    distance_rgb_teacher = kl_soft_dist(feat_rgb,feat_rgb)
                    _, rgb_teaching_loss_feat = Bg_kl(distance_rgb_student, distance_rgb_teacher)
                    _, rgb_teaching_loss_logit = Bg_kl(logit_student[sub == 0], logit_rgb)
                    loss_student += rgb_teaching_loss_feat
                    loss_student += rgb_teaching_loss_logit
                    metric.update({'rgb_teaching_loss_feat': rgb_teaching_loss_feat.data})
                    metric.update({'rgb_teaching_loss_logit': rgb_teaching_loss_logit.data})

                if self.IR_Teaching:
                    distance_ir_teacher = kl_soft_dist(feat_ir, feat_ir)
                    _, ir_teaching_loss_feat = Bg_kl(distance_ir_student, distance_ir_teacher)
                    _, ir_teaching_loss_logit = Bg_kl(logit_student[sub == 1], logit_ir)
                    loss_student += ir_teaching_loss_feat
                    loss_student += ir_teaching_loss_logit
                    metric.update({'ir_teaching_loss_feat': ir_teaching_loss_feat.data})
                    metric.update({'ir_teaching_loss_logit': ir_teaching_loss_logit.data})

        return loss_student, loss_rgb, loss_ir, metric





