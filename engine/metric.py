from collections import defaultdict

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, Accuracy


class ScalarMetric(Metric):
    # 一个自定义的指标类，用于在训练或评估过程中，对某些标量值进行累积和计算

    def update(self, value):
        self.sum_metric += value  # 累加传入的标量值。
        self.sum_inst += 1  # 记录累积的样本数量。

    def reset(self):
        # 重置
        self.sum_inst = 0
        self.sum_metric = 0

    def compute(self):
        if self.sum_inst == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        # 如果没有任何样本，抛出一个 NotComputableError，说明无法计算指标
        return self.sum_metric / self.sum_inst


class IgnoreAccuracy(Accuracy):
    # 一个自定义的准确率计算类
    def __init__(self, ignore_index=-1):
        # 参数 ignore_index，用于指定需要忽略的标签值（默认为 -1）。
        super(IgnoreAccuracy, self).__init__()

        self.ignore_index = ignore_index

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        # 处理每个 batch 的预测结果和真实标签，更新累积的正确预测数和样本总数。

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))  # 检查数据类型

        if self._type == "binary":
            # 二分类（binary）：对 y_pred 取四舍五入后的值作为预测类别（0 或 1）。
            indices = torch.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            # 多分类（multiclass）：对 y_pred 的每一行取最大值所在的索引（torch.max 返回最大值及其索引，dim=1 表示按列计算）。
            indices = torch.max(y_pred, dim=1)[1]

        correct = torch.eq(indices, y).view(-1)
        # 比较预测值 indices 和真实值 y
        ignore = torch.eq(y, self.ignore_index).view(-1)
        # 忽略样本
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0] - ignore.sum().item()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class AutoKVMetric(Metric):
    # 一个通用的（Key-Value）指标计算器
    # 它假设模型输出是一个 dict 类型的数据，其中每个键（Key）对应一个指标的名称，每个值（Value）是该指标的标量值。
    def __init__(self):
        self.kv_sum_metric = defaultdict(lambda: torch.tensor(0., device="cuda"))
        self.kv_sum_inst = defaultdict(lambda: torch.tensor(0., device="cuda"))

        self.kv_metric = defaultdict(lambda: 0)

        super(AutoKVMetric, self).__init__()

    def update(self, output):
        if not isinstance(output, dict):
            raise TypeError('The output must be a key-value dict.')

        for k in output.keys():
            self.kv_sum_metric[k].add_(output[k])
            self.kv_sum_inst[k].add_(1)

    def reset(self):
        for k in self.kv_sum_metric.keys():
            self.kv_sum_metric[k].zero_()
            self.kv_sum_inst[k].zero_()
            self.kv_metric[k] = 0

    def compute(self):
        for k in self.kv_sum_metric.keys():
            if self.kv_sum_inst[k] == 0:
                continue
                # raise NotComputableError('Accuracy must have at least one example before it can be computed')

            metric_value = self.kv_sum_metric[k] / self.kv_sum_inst[k]
            self.kv_metric[k] = metric_value.item()

        return self.kv_metric
