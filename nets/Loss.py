import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np

class WeightedMSELoss(nn.Module):
    def __init__(self, weights_map=None):
        super(WeightedMSELoss, self).__init__()
        self.weights_map = weights_map

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        if self.weights_map == None:
            weights = torch.ones_like(target)
        else:
            assert isinstance(self.weights_map, list)
            weights_map = np.array(self.weights_map, dtype=np.float32)
            index = target.cpu().numpy().round().astype(np.int32)
            weights = torch.from_numpy(weights_map[index-1]).to(target.device)

        loss = F.mse_loss(input, target, reduction='none')
        loss *= weights
        loss = loss.mean()
        return loss


class PLCCLoss(nn.Module):
    def __init__(self):
        super(PLCCLoss, self).__init__()

    def forward(self, preds, labels):
        # Ensure that preds and labels are tensors of the same shape.
        assert preds.shape == labels.shape

        mse_loss = nn.L1Loss()(preds, labels)
        # Calculate means
        mean_pred = torch.mean(preds)
        mean_label = torch.mean(labels)

        # Claculate Covariance
        dev_preds = preds - mean_pred
        dev_labels = labels - mean_label
        covariance = torch.mean(dev_preds * dev_labels)

        # Standard deviations
        std_dev_pred = torch.std(preds)
        std_dev_label = torch.std(labels)
        # Avoid division by zero
        if std_dev_pred == 0 or std_dev_label == 0:
            return torch.tensor(1., device=preds.device)

        # Pearson correlation coefficient
        corr = covariance / (std_dev_pred * std_dev_label)

        # Convert to loss: we want to minimize this value
        loss = 0.5 * (1 - corr)

        return loss + mse_loss


class MergedFidelityLoss(nn.Module):
    def __init__(self):
        super(MergedFidelityLoss, self).__init__()

    def forward(self, preds, labels):
        # Ensure that preds and labels are tensors of the same shape.
        assert preds.shape == labels.shape
        preds = preds.squeeze()
        labels = labels.squeeze()

        preds_even = preds[::2]
        preds_odd = preds[1::2]
        labels_even = labels[::2]
        labels_odd = labels[1::2]

        assert preds.ndim == 1 and labels.ndim == 1
        loss1 = nn.L1Loss()(preds_even, labels_even)
        # loss1 = nn.L1Loss()(preds, labels)

        pred_diff = preds_even - preds_odd
        gt_diff = labels_even - labels_odd
        g = 0.5 * (torch.sign(gt_diff) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(pred_diff / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)
        esp = 1e-12
        loss2 = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))
        # print(loss1, loss2)
        loss = loss1 + loss2
        return loss


class MergedRankingLoss(nn.Module):
    def __init__(self):
        super(MergedRankingLoss, self).__init__()

    def forward(self, preds, labels):
        # Ensure that preds and labels are tensors of the same shape.
        assert preds.shape == labels.shape
        preds = preds.squeeze()
        labels = labels.squeeze()

        preds_even = preds[::2]
        preds_odd = preds[1::2]
        labels_even = labels[::2]
        labels_odd = labels[1::2]

        assert preds.ndim == 1 and labels.ndim == 1
        loss1 = nn.L1Loss()(preds_even, labels_even)

        pred_diff = preds_even - preds_odd

        l_ = torch.tensor([torch.exp(-i) if i < 0 else 0 for i in pred_diff])

        loss2 = torch.mean(l_)

        # print(loss1, loss2)
        loss = loss1 + loss2
        return loss


class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, preds, labels):
        cols_preds = torch.unbind(preds, dim=1)
        cols_labels = torch.unbind(labels, dim=1)
        loss_list = [self.pairwise_rank(preds, labels) for preds, labels in zip(cols_preds, cols_labels)]
        loss = torch.mean(torch.stack(loss_list))
        return loss

    def pairwise_rank(self, preds, labels):
        preds = preds.unsqueeze(1)
        labels = labels.unsqueeze(1)
        assert preds.ndim == 2 and preds.size(1) == 1  # assert Nx1
        assert labels.ndim == 2 and labels.size(1) == 1  # assert Nx1
        preds_diffs = preds - preds.t()
        labels_diffs = labels - labels.t()
        sign_errors = torch.relu(-(preds_diffs * labels_diffs))
        loss = torch.mean(sign_errors)
        return loss


class MSERankLoss(RankLoss):
    def __init__(self):
        super(MSERankLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, labels):
        cols_preds = torch.unbind(preds, dim=1)
        cols_labels = torch.unbind(labels, dim=1)
        loss_list = [self.pairwise_rank(preds, labels) for preds, labels in zip(cols_preds, cols_labels)]
        rank_loss = torch.mean(torch.stack(loss_list))
        mse_loss = self.mse(preds, labels)
        return 400 * rank_loss + mse_loss


class FidelityLoss(nn.Module):
    """prediction monotonicity related loss"""

    def __init__(self):
        super(FidelityLoss, self).__init__()

    def forward(self, y_pred, y):
        esp = 1e-8
        assert y_pred.size(0) > 1
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()
        # calculate g as the binary label for each image pair
        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        # calculate p by the standard normal culmulative distribution function
        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        # fidelity loss
        loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

        return loss


class TwoDirectionRankLoss(nn.Module):
    """two direction ranking loss"""

    def __init__(self):
        super(TwoDirectionRankLoss, self).__init__()

    def rank_loss(self, p, n):
        # calculate loss for the neighbor class
        l = -1.0 * torch.log(torch.exp(p) / (torch.exp(p) + torch.exp(n)))
        return l

    def forward(self, y_pred, y):
        assert y_pred.size(0) > 1  #
        loss = 0
        for pred, gt in zip(y_pred, y):
            # get rounded gt as idx supposed to be the highest class
            idx = torch.round(gt).long() - 1
            # print(idx, gt)
            # leftward ranking
            for i in range(1, idx):
                loss += self.rank_loss(pred[i], pred[i - 1])
            # rightward ranking
            for i in range(idx, 4):
                loss += self.rank_loss(pred[i], pred[i + 1])
        loss = loss / y_pred.size(0)

        return loss

class FaClipLoss(nn.Module):
    def __init__(self):
        super(FaClipLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.fidelity_loss = FidelityLoss()
        self.ranking_loss = TwoDirectionRankLoss()
    def forward(self, preds, labels):
        assert isinstance(preds, tuple), 'preds must be a tuple'
        assert isinstance(labels, torch.Tensor), 'labels must be a torch.Tensor'

        l1_loss = self.l1_loss(preds[0], labels)
        fidelity_loss = self.fidelity_loss(preds[0], labels)
        ranking_loss = self.ranking_loss(preds[1], labels)
        loss = l1_loss + fidelity_loss + ranking_loss
        return loss


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, preds, labels):
        """
        Earth Mover's Distance (EMD) loss
        参数:
        y_true -- 真实分布张量, 大小为 (batch_size, num_bins)
        y_pred -- 预测分布张量, 大小为 (batch_size, num_bins)
        """
        # 计算累计分布函数 (CDF)
        cdf_true = torch.cumsum(labels, dim=1)
        cdf_pred = torch.cumsum(preds, dim=1)
        
        # 计算两个CDF之间的绝对差
        cdf_diff = torch.abs(cdf_true - cdf_pred)
        
        # 对所有bin的差进行求和，获取批次的平均
        emd_loss = torch.mean(torch.sum(cdf_diff, dim=1))
        
        return emd_loss