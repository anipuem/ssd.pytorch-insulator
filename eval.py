"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import struct

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_VOC_11999.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.6, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--show_results', default=True, type=str2bool,
                    help='show image detection results after detection')

args, unknow= parser.parse_known_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', 'test.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)  #RGB均值
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """读取xml文件上的选框标记"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        if obj.find('difficult'):
            obj_struct['difficult'] = int(obj.find('difficult').text)
        else:
            obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath, cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('----------------------------')
    print('Results of mAP is:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('----------------------------')


def voc_ap(rec, prec, use_07_metric=True):  # https://blog.csdn.net/weixin_43646592/article/details/113998328
    """ 通过准确率和召回率计算AP"""
    if use_07_metric:  # 11 point metric
        ap = 0.
        # 11点法，就是将结果先按置信度顺序排序，然后分别将recall大于0，大于0.1，大于0.2的…大于1的数据找出来（共11组）
        # 然后分别取这11组数据中的precision最大的找出来，这时，我们将会得到11个precision值，sum所有（recall对应precision）
        for t in np.arange(0, 1.1, 0.1):  # 起点为0，终点为1.1，步长为0.1
            if np.sum(rec >= t) == 0:
                p = 0  # rec=0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 将recall和precision补全，主要用于积分计算，保证recall的域为[0,1]
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # 滤除fp增加条件下导致的pre减小的无效值
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # 滤除总检测样本数增加导致计算的recall的未增加的量
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # 通过积分计算precision对recall的平均数： sum (\Delta recall) * precision
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,  # Path to detections, detpath.format(classname) should produce the detection results file
             annopath,  # Path to annotations
             imagesetfile,  # .txt file containing the list of images's name, one image per line
             classname,
             cachedir,  # 缓存模型检测到的annotations的目录
             ovthresh=0.8,  # Overlap threshold (default = 0.5)
             use_07_metric=True):  # 是否用voc数据集11点法求AP
    """该函数会对比GT和result，来获得Output = rec, prec, ap"""
    """获取GT：（人工标注目标）"""
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)  # 如果目前没有，则创建缓存目录
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # 读取test的图像名称
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]  # .strip()用于去除lines中每行首尾的空格

    # 根据imagenames列表读取-->(else)/存储-->(if not)xml标签信息
    if not os.path.isfile(cachefile):
        # 载入标签文件：recs字典中，存储了验证集所有的GT信息(Ground Truth-->人工标注为绝缘子的选框)
        recs = {}
        for imagename in imagenames:  # recs{imagename：对应图像的xml内容}
            recs[imagename] = parse_rec(r'./data/VOCdevkit/VOC2007/Annotations/'+imagename+'.xml')
        # 将读取的xml内容存入缓存文件annots.pkl
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:  # 读取recs{imagename：对应图像的xml内容}，因为对于相同验证集图片，recs是不变的
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # 从字典recs中提取当前类型（insulator）的GT标签信息
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # obj_name是classname的全都存入R
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])  # bbox存储该文件中所有box信息
        # difficult含义：如果为1的话，表示难检测，所以=1时模型检测不出来，也不会当做漏检
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  # 转换成布尔变量[0,1]
        det = [False] * len(R)
        # 统计xml文件中<difficult>0</difficult>的个数（即~difficult=1，~取反），用来表示gt的个数
        npos = npos + sum(~difficult)
        # class_recs是一个字典，第一层key为文件名，一个文件名对应的子字典中，
        # 存储了key对应的图片文件中所有的该类型的box、difficult、det信息（信息中可以有多个GT）
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}  #
    """获取result：（检测结果）"""
    # 从当前class的result文件中读取结果，并将结果按照confidence从大到小排序
    detfile = detpath.format(classname)  # detfile == the detection results file
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:  # 避免文件为空
        splitlines = [x.strip().split(' ') for x in lines]
        # 因为会有很多每行都是名称+置信度+BB，所以以下这三者都是一一对应的
        # 图像名称，因为图像命名为string (num)，所以0-1都是名称
        image_ids = [x[0]+' '+x[1] for x in splitlines]
        # 提取每个结果的置信度，存入confidence
        confidence = np.array([float(x[2]) for x in splitlines])
        # 提取每个结果的box选框，以list的形式存入BB
        BB = np.array([[float(z) for z in x[3:]] for x in splitlines])

        # 按confidence从大到小（argsort()是从小到大）排序，获取排序后索引的list
        sorted_ind = np.argsort(-confidence)  # sorted_scores = np.sort(-confidence)-->可以得到从大到小排序的置信度list
        BB = BB[sorted_ind, :]  # 按confidence排序对BB进行排序，（n,4）的矩阵
        image_ids = [image_ids[x] for x in sorted_ind]  # 按confidence对相应的图像的id进行排序
        print(BB)  # [[135.1 151.5 198.8 400. ] [252.8 214.5 294.9 367.5]]这样子
        print(BB.shape)  # (5797, 4)
        """在这里考虑通过黑色像素占比消除部分选框"""

        # 对比GT参数和result，计算出IOU，在fp和tp相应位置标记1
        nd = len(image_ids)
        # 初始化tp和fp都为0，之后判断类型了在变为1，最后只计算tp和fp的累加值
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        # 对一个result文件中所有目标进行遍历
        for d in range(nd):  # d：模型检测到的bbox
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)  # 将BB中confidence第d大的BB内容提取到bb中，这是result中的信息
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)  # 当前confidence从大到小排序条件下，第d个GT中bbox中的信息
            # 如果BBGT中有信息，即该图片中有目标-->计算IOU
            if BBGT.size > 0:
                # 计算intersection的四个坐标值
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                # 算w和h-->大于0就输出正常值，小于等于0就输出0
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                # I=重合面积
                inters = iw * ih
                # U=union总面积
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)  # inter被算了两次所以-inters
                # 计算ov = 重叠部分面积 / 联合的面积，并记录ovmax, jmax
                overlaps = inters / uni
                ovmax = np.max(overlaps)  # 选出最大交并比
                jmax = np.argmax(overlaps)  # 求出两个最大交并比的值的GT序号
            # ovmax原始值为-inf，则没有目标肯定不可能进入if下面的任务
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:  # j表示图像中的GroundTruth框
                    if not R['det'][jmax]:  # difficult和det初始值都是False
                        tp[d] = 1.  # true-positive正样本检测为真
                        R['det'][jmax] = 1  # 标记为已检测
                    else:
                        tp[d] = 1.  # 被检测了两次
                    """一个GT被用到两次的时候！！这里考虑加nms"""
            else:
                fp[d] = 1.  # 交并比未达到要求，认为是错误检测框
                """考虑删除选框，因为是错误检测"""

        """计算ap,rec，prec："""
        fp = np.cumsum(fp)  # 采用cumsum计算结果是一种积分形式的累加序列，假设fp=[0,1,1],那么np.cumsum(fp)为[0,1,2]
        tp = np.cumsum(tp)  # 可以把每一次的结果0，1，0，1，1变成累加结果0，1，1，2，3
        # 不仅要衡量检测出正确目标的数量，还应该评价模型是否能以较高的precision检测出目标
        # 也就是在某个类别下的检测，在检测出正确目标之前，是不是出现了很多判断失误。AP越高，说明检测失误越少
        # ---------- 实际 --------------
        # |        正            负                  TP                       TP
        # 预  正   TP            FP       Recall=-----------，  Precision=-----------
        # 测  负   FN            TN                 TP+FN                   TP+FP
        # |-----------------------------
        rec = tp / float(npos)  # 召回率：实际为正样本中，预测为正样本数，npos表示gt的个数
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 准确率:预测为正样本中，实际为正样本数，np.finfo(np.float64).eps防止分母为0
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
        print(str(detfile)+' is empty')
    return rec, prec, ap


def test_net(net, dataset):  # 将测试数据集送入net进行推断出来detections
    # save_folder, net, cuda, dataset, transform, top_k, im_size=300, thresh=0.05
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)  # 返回单张图像及其标签
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)
        diff_cls_dets = []  # 用来在图片上画框
        for j in range(1, detections.size(1)):  # 因为j=0是背景类，所以从j=1开始
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()  # softmax之后只有target和not target两类
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            # clean up the dets
            num_box = cls_dets.shape[0]
            list_clean_up = []
            for each_box_index in range(num_box):
                each_box = cls_dets[each_box_index]
                if each_box[4] < 0.8:
                    list_clean_up.append(each_box_index)
            cls_dets = np.delete(cls_dets, list_clean_up, axis=0)
            all_boxes[j][i] = cls_dets
            diff_cls_dets.append(np.array(cls_dets))

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
        if args.show_results:  # 在图片上画出检测框
            img = dataset.pull_image(i)
            for color_num, diff_cls_det in enumerate(diff_cls_dets):
                for k in range(diff_cls_det.shape[0]):
                    x0 = diff_cls_det[k, 0]
                    y0 = diff_cls_det[k, 1]
                    x1 = diff_cls_det[k, 2]
                    y1 = diff_cls_det[k, 3]
                    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.imwrite("./eval/" + str(dataset.ids[i]).split(r'\\')[-1].split('\'')[-2] + '.png', img)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)  # all_boxes是检测框+是否是target（0，1）


def evaluate_detections(box_list, output_dir, dataset):  # 不改变参数
    write_voc_results_file(box_list, dataset)  # 写result
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # labelmap里面是所有类别，+1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))  # 将预训练的参数权重加载到新的模型
    net.eval()
    print('Finished loading model!')
    # 加载数据 dataset是个class对象，存储了各类信息
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, dataset_mean),
                           VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(net,
             dataset
             )  # need to be bytes



