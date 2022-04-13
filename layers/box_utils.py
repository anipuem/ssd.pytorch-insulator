# -*- coding: utf-8 -*-
import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


# truths = targets[idx][:, :-1].data  # 取出target中的坐标 [num_objs, 4]
# defaults = priors.data              # 先验框anchors  [8732, 4]
# variances                           # [0.1, 0.2]
# labels = targets[idx][:, -1].data   # 取出target中每个目标实体对应的类别 [batch_size, num_objs]
# loc_t                               # [batch_size, 8732, 4]
# conf_t                              # [batch_size, 8732]
# idx                                 # batch数据的index
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

    # print(truths.shape, priors.shape, labels.shape, loc_t.shape, conf_t.shape)
    # print(truths)
    # print(labels)

    """
    (22, 4) (8732, 4) (22,) (2, 8732, 4) (2, 8732)
    tensor([[0.5410, 0.3665, 0.5615, 0.4030],
            [0.5648, 0.3678, 0.5726, 0.4033],
            [0.2499, 0.5137, 0.2587, 0.5307],
            [0.2630, 0.5054, 0.2691, 0.5200],
            [0.2125, 0.5057, 0.2210, 0.5340],
            [0.5635, 0.4664, 0.6070, 0.6341],
            [0.5416, 0.4639, 0.5682, 0.5646],
            [0.4424, 0.4675, 0.4740, 0.5798],
            [0.4692, 0.4830, 0.4768, 0.5564],
            [0.4359, 0.4692, 0.4540, 0.5672],
            [0.4130, 0.4774, 0.4377, 0.5895],
            [0.3746, 0.4907, 0.3873, 0.5456],
            [0.3740, 0.4784, 0.3857, 0.4990],
            [0.3511, 0.4749, 0.3743, 0.5662],
            [0.4899, 0.4929, 0.5432, 0.5268],
            [0.3528, 0.4882, 0.3678, 0.5301],
            [0.2655, 0.4871, 0.2800, 0.5418],
            [0.2384, 0.4906, 0.2557, 0.5616],
            [0.3864, 0.4833, 0.4070, 0.5620],
            [0.2118, 0.4871, 0.2152, 0.4967],
            [0.3121, 0.4748, 0.3367, 0.5665],
            [0.2064, 0.2220, 0.2064, 0.2220]])
    tensor([74., 74., 26., 26., 26.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            13., 26.,  0.,  0.,  0., 24.,  0.,  0.])
            
    (2, 4) (8732, 4) (2,) (2, 8732, 4) (2, 8732)
    tensor([[0.3050, 0.9739, 0.3421, 1.0000],
            [0.3560, 0.9702, 0.4255, 1.0000]])
    tensor([18., 18.])
    """

    # jaccard相似系数(IOU)：计算目标实体框与先验框anchors的IOU shape: [box_a.size(0), box_b.size(0)] => [num_objs, 8732]
    overlaps = jaccard(
        truths,             # shape: (num_objs, 4)
        point_form(priors)  # (x,y,w,h) => (x1,y1, x2,y2) shape: (8732, 4)
    )
    # (Bipartite Matching)
    # 获取每个目标实体框对应的最优先验框anchor shape: [1, num_objs]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # 获取每个先验框对应的最优目标实体框truth  shape: [1, 8732]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # best_truth_overlap保存了所有先验框对应的最优ground_truth的IOU值
    # IOU值的范围是[0, 1]，为了突出针对ground_truth的最优anchors，将最优anchors的IOU值设置为2
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior


    # TODO refactor: index  best_prior_idx with long tensor
    # 再次确认，最优anchors对应了最优ground_truth, 一般情况下best_truth_idx[best_prior_idx[j]]等于j，这一步只是再次确认。
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 每个先验框anchor对应目标ground_truth的真实坐标 Shape: [8732, 4]
    matches = truths[best_truth_idx]
    # 每个先验框anchor对应目标ground_truth的类别标签 Shape: [8732],
    # 注意，这里需要加1，因为预测的类别中第0位是置信度，类别是从1开始计数。
    conf = labels[best_truth_idx] + 1

    # 把IOU小于阈值的设置为非置信先验框，
    # 从这里可以看出conf有两种功能，当conf值等于0时表示置信度，代表非置信；大于0时表示类别，也就默认置信了。
    conf[best_truth_overlap < threshold] = 0  # label as background
    # 根据ground_truth坐标信息和先验框坐标信息，得到先验框坐标转为ground_truth坐标的偏移系数（offsets）
    loc = encode(matches, priors, variances)

    # 该流程最终得到了当前图像中，先验框转为真实坐标的偏移系数（loc_t）和先验框的置信度/类别（conf_t）
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    # print('loc: ' + loc + ', and conf: ' + conf)


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh + 1e-10) / variances[1]   # 为了避免log0出现
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max  # 正则化->减掉最大的x_max，避免exp()=inf


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        try:
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
        except:
            xx1 = torch.index_select(x1, 0, idx)
            yy1 = torch.index_select(y1, 0, idx)
            xx2 = torch.index_select(x2, 0, idx)
            yy2 = torch.index_select(y2, 0, idx)
            xx1 = torch.clamp(xx1, min=x1[i].data)
            yy1 = torch.clamp(yy1, min=y1[i].data)
            xx2 = torch.clamp(xx2, max=x2[i].data)
            yy2 = torch.clamp(yy2, max=y2[i].data)
        w.resize_as_(xx2)
        h.resize_as_(yy2)  # h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
