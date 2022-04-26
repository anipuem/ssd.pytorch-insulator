#!/usr/bin/env python
# encoding: utf-8

"""
@version: 
@author: Asn
@file: boxCheck.py
@time: 2022/4/26 19:43
"""

import cv2
import os
from xml.etree import ElementTree as ET
import random
import shutil


def read_(path):
    # 检测的类别需要修改，根据自己检测的类别名修改即可
    voc_name_list = ['insulator']
    with open(r'./del.txt', 'w') as writer:
        root_dir = path
        dir_path = os.path.join(root_dir, "JPEGImages")
        xml_dir_path = os.path.join(root_dir, "Annotations")
        files = os.listdir(xml_dir_path)
        random.shuffle(files)
        for name in files:
            xml_file = os.path.join(xml_dir_path, name)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = root.find("filename").text
            writer.writelines(filename.split('.')[0] + ' ')
            image_file_path = os.path.join(dir_path, filename)
            writer.writelines(image_file_path)
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find("bndbox")
                xmin = int(float(bbox.find("xmin").text))
                ymin = int(float(bbox.find("ymin").text))
                xmax = int(float(bbox.find("xmax").text))
                ymax = int(float(bbox.find("ymax").text))
                if xmin < 0:
                    xmin =0
                if ymin < 0:
                    ymin =0

                writer.writelines(" {},{},{},{},{}".format(xmin, ymin, xmax, ymax, voc_name_list.index(label)))
            writer.writelines('\n')


def draw_result(info_):
    if os.path.exists('./test'): shutil.rmtree('./test')
    if not os.path.exists('./test'):
        os.mkdir('./test')
    # 检测的类别需要修改，根据自己检测的类别名修改即可
    voc_name_list = ['insultaor']
    ii = 0
    bbox_nums = []
    for i in info_:
        ii += 1
        x = i.split(' ')
        imgname = x[0]+' '+x[3]
        img = cv2.imread('./origin_img/JPEGImages/'+imgname)
        for j in range(len(x[4:])):
            bbox_ = x[j+4]
            bbox = bbox_.split(',')[:-1]  # eg.['46', '131', '162', '158']
            bbox_nums.append(bbox)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            # labels = bbox_.split(',')[-1]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(
            #     img, '%s' % voc_name_list[int(float(labels))],
            #     (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
            #     (0, 255, 0), 2, 8)
        if img is None:
            print(x[0]+' '+x[3])
        cv2.imwrite('./test/{}'.format(x[0]+' '+x[3]), img)
        print('Image '+ x[0]+' '+x[3]+' done')


if __name__ == '__main__':
    # 存放img和xml的文件夹
    path = './origin_img/'
    read_(path)

    with open(r'./del.txt', 'r') as writer:
        txt = writer.readlines()
    txt1 = []
    for x in txt:
        if x == '\n':
            continue
        else:
            txt1.append(x.strip())
    draw_result(txt1)
