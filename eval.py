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
dataset_mean = (104, 117, 123)  #RGB??????
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
    """??????xml????????????????????????"""
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
    """ ?????????????????????????????????AP"""
    if use_07_metric:  # 11 point metric
        ap = 0.
        # 11?????????????????????????????????????????????????????????????????????recall??????0?????????0.1?????????0.2????????????1????????????????????????11??????
        # ??????????????????11???????????????precision????????????????????????????????????????????????11???precision??????sum?????????recall??????precision???
        for t in np.arange(0, 1.1, 0.1):  # ?????????0????????????1.1????????????0.1
            if np.sum(rec >= t) == 0:
                p = 0  # rec=0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # ???recall???precision??????????????????????????????????????????recall?????????[0,1]
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # ??????fp????????????????????????pre??????????????????
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # ?????????????????????????????????????????????recall??????????????????
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # ??????????????????precision???recall??????????????? sum (\Delta recall) * precision
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,  # Path to detections, detpath.format(classname) should produce the detection results file
             annopath,  # Path to annotations
             imagesetfile,  # .txt file containing the list of images's name, one image per line
             classname,
             cachedir,  # ????????????????????????annotations?????????
             ovthresh=0.8,  # Overlap threshold (default = 0.5)
             use_07_metric=True):  # ?????????voc?????????11?????????AP
    """??????????????????GT???result????????????Output = rec, prec, ap"""
    """??????GT???????????????????????????"""
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)  # ??????????????????????????????????????????
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # ??????test???????????????
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]  # .strip()????????????lines????????????????????????

    # ??????imagenames????????????-->(else)/??????-->(if not)xml????????????
    if not os.path.isfile(cachefile):
        # ?????????????????????recs???????????????????????????????????????GT??????(Ground Truth-->?????????????????????????????????)
        recs = {}
        for imagename in imagenames:  # recs{imagename??????????????????xml??????}
            recs[imagename] = parse_rec(r'./data/VOCdevkit/VOC2007/Annotations/'+imagename+'.xml')
        # ????????????xml????????????????????????annots.pkl
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:  # ??????recs{imagename??????????????????xml??????}???????????????????????????????????????recs????????????
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # ?????????recs????????????????????????insulator??????GT????????????
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # obj_name???classname???????????????R
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])  # bbox????????????????????????box??????
        # difficult??????????????????1?????????????????????????????????=1????????????????????????????????????????????????
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  # ?????????????????????[0,1]
        det = [False] * len(R)
        # ??????xml?????????<difficult>0</difficult>???????????????~difficult=1???~????????????????????????gt?????????
        npos = npos + sum(~difficult)
        # class_recs???????????????????????????key??????????????????????????????????????????????????????
        # ?????????key?????????????????????????????????????????????box???difficult???det?????????????????????????????????GT???
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}  #
    """??????result?????????????????????"""
    # ?????????class???result??????????????????????????????????????????confidence??????????????????
    detfile = detpath.format(classname)  # detfile == the detection results file
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:  # ??????????????????
        splitlines = [x.strip().split(' ') for x in lines]
        # ????????????????????????????????????+?????????+BB?????????????????????????????????????????????
        # ????????????????????????????????????string (num)?????????0-1????????????
        image_ids = [x[0]+' '+x[1] for x in splitlines]
        # ???????????????????????????????????????confidence
        confidence = np.array([float(x[2]) for x in splitlines])
        # ?????????????????????box????????????list???????????????BB
        BB = np.array([[float(z) for z in x[3:]] for x in splitlines])

        # ???confidence???????????????argsort()???????????????????????????????????????????????????list
        sorted_ind = np.argsort(-confidence)  # sorted_scores = np.sort(-confidence)-->??????????????????????????????????????????list
        BB = BB[sorted_ind, :]  # ???confidence?????????BB??????????????????n,4????????????
        image_ids = [image_ids[x] for x in sorted_ind]  # ???confidence?????????????????????id????????????
        # print(BB)  # [[135.1 151.5 198.8 400. ] [252.8 214.5 294.9 367.5]]?????????
        # print(BB.shape)  # (5797, 4)
        """?????????????????????????????????????????????????????????"""

        # ??????GT?????????result????????????IOU??????fp???tp??????????????????1
        nd = len(image_ids)
        # ?????????tp???fp??????0?????????????????????????????????1??????????????????tp???fp????????????
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        # ?????????result?????????????????????????????????
        for d in range(nd):  # d?????????????????????bbox
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)  # ???BB???confidence???d??????BB???????????????bb????????????result????????????
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)  # ??????confidence?????????????????????????????????d???GT???bbox????????????
            # ??????BBGT???????????????????????????????????????-->??????IOU
            if BBGT.size > 0:
                # ??????intersection??????????????????
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                # ???w???h-->??????0?????????????????????????????????0?????????0
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                # I=????????????
                inters = iw * ih
                # U=union?????????
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)  # inter?????????????????????-inters
                # ??????ov = ?????????????????? / ???????????????????????????ovmax, jmax
                overlaps = inters / uni
                ovmax = np.max(overlaps)  # ?????????????????????
                jmax = np.argmax(overlaps)  # ????????????????????????????????????GT??????
            # ovmax????????????-inf???????????????????????????????????????if???????????????
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:  # j??????????????????GroundTruth???
                    if not R['det'][jmax]:  # difficult???det???????????????False
                        tp[d] = 1.  # true-positive?????????????????????
                        R['det'][jmax] = 1  # ??????????????????
                    else:
                        tp[d] = 1.  # ??????????????????
                    """??????GT?????????????????????????????????????????????nms"""
            else:
                fp[d] = 1.  # ???????????????????????????????????????????????????
                """??????????????????????????????????????????"""

        """??????ap,rec???prec???"""
        fp = np.cumsum(fp)  # ??????cumsum?????????????????????????????????????????????????????????fp=[0,1,1],??????np.cumsum(fp)???[0,1,2]
        tp = np.cumsum(tp)  # ???????????????????????????0???1???0???1???1??????????????????0???1???1???2???3
        # ??????????????????????????????????????????????????????????????????????????????????????????precision???????????????
        # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????AP?????????????????????????????????
        # ---------- ?????? --------------
        # |        ???            ???                  TP                       TP
        # ???  ???   TP            FP       Recall=-----------???  Precision=-----------
        # ???  ???   FN            TN                 TP+FN                   TP+FP
        # |-----------------------------
        rec = tp / float(npos)  # ????????????????????????????????????????????????????????????npos??????gt?????????
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # ?????????:????????????????????????????????????????????????np.finfo(np.float64).eps???????????????0
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
        print(str(detfile)+' is empty')
    return rec, prec, ap


def test_net(net, dataset):  # ????????????????????????net??????????????????detections
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
        im, gt, h, w = dataset.pull_item(i)  # ??????????????????????????????
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)
        diff_cls_dets = []  # ????????????????????????
        for j in range(1, detections.size(1)):  # ??????j=0????????????????????????j=1??????
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
            scores = dets[:, 0].cpu().numpy()  # softmax????????????target???not target??????
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            # clean up the dets
            list_clean_up = []
            # impath = r'./data/VOCdevkit/VOC2007/JPEGImages/'+str(dataset.ids[i]).
            # split(r'\\')[-1].split('\'')[-2] + '.png'
            # img = cv2.imread(impath)
            for k in range(cls_dets.shape[0]):
                img = dataset.pull_image(i)
                xmin = cls_dets[k, 0]
                ymin = cls_dets[k, 1]
                xmax = cls_dets[k, 2]
                ymax = cls_dets[k, 3]
                # ???????????????[y0:y1, x0:x1]
                if int(min(ymax, h))-int(max(ymin, 0)) < 5 and int(min(xmax, w))-int(max(xmin, 0)) < 5:
                    list_clean_up.append(k)
                    continue
                img = img[int(max(ymin, 0)):int(min(ymax, h)), int(max(xmin, 0)):int(min(xmax, w))]
                npim = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
                # ?????????3?????????????????????????????????
                npim[:] = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
                # ????????????????????????
                black = len(npim[npim == 0])
                if black > img.shape[0]*img.shape[1] * 0.75:
                    list_clean_up.append(k)
            # numpy.delete(arr,obj,axis=None) obj??????????????????????????????????????????int???list?????????
            cls_dets = np.delete(cls_dets, list_clean_up, axis=0)
            all_boxes[j][i] = cls_dets
            diff_cls_dets.append(np.array(cls_dets))

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
        if args.show_results:  # ???????????????????????????
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
    evaluate_detections(all_boxes, output_dir, dataset)  # all_boxes????????????+?????????target???0???1???


def evaluate_detections(box_list, output_dir, dataset):  # ???????????????
    write_voc_results_file(box_list, dataset)  # ???result
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # labelmap????????????????????????+1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))  # ????????????????????????????????????????????????
    net.eval()
    print('Finished loading model!')
    # ???????????? dataset??????class??????????????????????????????
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



