import torch
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())  # 报错提示过时，更新为如下
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        # y = dist_an.data.new()
        # y.resize_as_(dist_an.data)
        # y.fill_(1)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = dist_an.data > dist_ap.data
        length = torch.sqrt((inputs * inputs).sum(1)).mean()
        return loss, dist,dist_ap, dist_an
