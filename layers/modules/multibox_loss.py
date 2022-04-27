# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function 论文：https://arxiv.org/pdf/1512.02325.pdf
    Compute Targets:
        1) 通过匹配GT和prior box生成目标置信度目录，取jaccard index>threshold 的prior box(默认阈值:0.5)
        2) encoding将ground truth做成prior的格式，variance是放大梯度使模型训练加速
        3) 进行难例挖掘过滤default bounding boxes生成的过多的反例，否则正负样本会失衡(默认负:正=3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N  --> weighted by α which is set to 1 by cross val.
        Where Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        Args:
            c: class confidences
            l: predicted boxes
            g: ground truth boxes
            N: number of matched default boxes
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = 2  # 要加背景类
        self.threshold = overlap_thresh  # 0.5
        self.background_label = bkg_label  # 背景类对应的label编号，默认0
        self.encode_target = encode_target  # False
        self.use_prior_for_matching = prior_for_matching  # True
        self.do_neg_mining = neg_mining  # True
        self.negpos_ratio = neg_pos  # 负正样本比例，默认3：1 == 3
        self.neg_overlap = neg_overlap  # 0.5
        self.variance = cfg['variance']  # # 梯度放缩，训练加速[0.1, 0.2]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds，conf preds and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)  4是四个角坐标
        Targets (tensor): Ground truth boxes and labels for one batch,
                shape: [batch_size,num_objs,5] 5因为除了四个角坐标还有类别，类别在最后.
        """
        # (1, 8732, 4) (1, 8732, 2) (8732, 4)
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)  # batch_size
        priors = priors[:loc_data.size(1), :]  # 取出8732个锚点框,与位置预测的锚点框数量相同
        num_priors = priors.shape[0]  # num_anchors
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)    # 坐标有4个参数
        conf_t = torch.LongTensor(num, num_priors)  # 置信度只有1参数，每个anchor对应一个置信度
        for idx in range(num):  # 对batch数据，单条依次处理
            try:  # 为了避免图像中没有目标，所以GT为空的情况
                truths = targets[idx][:, :-1].data  # 取出target中的GT坐标 [num_objs, 4]
                labels = targets[idx][:, -1].data   # 取出target中每个目标实体对应的类别 [num_objs]
                defaults = priors.data              # 先验框anchors  [8732, 4]
                # 该流程最终得到了当前图像中，先验框转为真实坐标的偏移系数（loc_t）和先验框的置信度/类别（conf_t）
                match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            except NameError as e:
                print('此图片没有GT：'+e)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 封装
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0  # 当conf_t大于0时，统计正样本数
        num_pos = pos.sum(dim=1, keepdim=True)  # 计算置信先验框的个数

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)  # 置信先验框转为ground_truth预测prediction的偏移系数
        loc_t = loc_t[pos_idx].view(-1, 4)     # 置信先验框转为ground_truth真实truth的偏移系数
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # 偏移系数loss

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # 难例挖掘中要过滤掉正样本
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # 计算正负置信样例，正负样按比例取，保证样本均衡
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum().float()  # 置信先验框的个数
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
