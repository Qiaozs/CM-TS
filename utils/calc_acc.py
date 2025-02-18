import torch


def calc_acc(logits, label, ignore_index=-100, mode="multiclass"):
    """
    输入：
    logits 预测值
    label 标签
    """
    if mode == "binary":  # 二分类
        indices = torch.round(logits).type(label.type())
    elif mode == "multiclass":  # 多分类

        indices = torch.max(logits, dim=1)[1]  # 取 logits 最大值的索引

    if label.size() == logits.size():
        ignore = 1 - torch.round(label.sum(dim=1))  # 忽略全零行（假设无效样本）

        # torch.round()将输入张量的每个元素舍入到最近的整数。
        label = torch.max(label, dim=1)[1]

    else:
        ignore = torch.eq(label, ignore_index).view(-1)

    correct = torch.eq(indices, label).view(-1)

    num_correct = torch.sum(correct)
    num_examples = logits.shape[0] - ignore.sum()

    return num_correct.float() / num_examples.float()
# def calc_acc(logits, label, ignore_index=-100, mode="multiclass"):
#     """
#     输入：
#     logits 预测值
#     label 标签
#     """
#     print("label:", label)
#     print("mode:", mode)
#     print("logits:", logits)
#     if mode == "binary":  # 二分类
#         indices = torch.round(logits).type(label.type())
#     elif mode == "multiclass":  # 多分类
#
#         indices = torch.max(logits, dim=1)[1]  # 取 logits 最大值的索引
#         print("indices={}".format(indices))
#
#     print(label.size(), logits.size())
#     if label.size() == logits.size():
#         print("true")
#         ignore = 1 - torch.round(label.sum(dim=1))  # 忽略全零行（假设无效样本）
#         print("ignore:",ignore)
#
#         # torch.round()将输入张量的每个元素舍入到最近的整数。
#         label = torch.max(label, dim=1)[1]
#         print("label:",label)
#
#     else:
#         ignore = torch.eq(label, ignore_index).view(-1)
#         print("ignore:",ignore)
#
#     correct = torch.eq(indices, label).view(-1)
#     print("correct:", correct)
#     num_correct = torch.sum(correct)
#     num_examples = logits.shape[0] - ignore.sum()
#     print("num_examples:", num_examples)
#     print(num_correct.float() / num_examples.float())
#     return num_correct.float() / num_examples.float()