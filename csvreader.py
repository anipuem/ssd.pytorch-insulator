#!/usr/bin/env python
# encoding: utf-8

"""
@version: 
@author: Asn
@file: csvreader.py
@time: 2021/3/15 11:19
"""

import csv
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
filename = 'Gray_good.csv'

with open(filename, 'r', encoding="utf-8") as f:
    reader = csv.reader(f)
    graynum_row = next(reader)
    fault1 = next(reader)
    fault1 = [float(x) for x in fault1]
    fault2 = next(reader)
    fault2 = [float(x) for x in fault2]
    fault3 = next(reader)
    fault3 = [float(x) for x in fault3]
    fault4 = next(reader)
    fault4 = [float(x) for x in fault4]
    fault5 = next(reader)
    fault5 = [float(x) for x in fault5]
    fault6 = next(reader)
    fault6 = [float(x) for x in fault6]
    fault7 = next(reader)
    fault7 = [float(x) for x in fault7]
    fault8 = next(reader)
    fault8 = [float(x) for x in fault8]
    fault9 = next(reader)
    fault9 = [float(x) for x in fault9]

def f(x):
    total = np.sum(x)
    step = 8
    judge = 0
    b = [np.sum(x[i:i + step]) for i in range(0, len(x), step)]  # 八个一组切片
    a = []
    for i in range(len(x)):
        if i == 0:
            new = x[i]
        else:
            new = x[i] + a[i-1]
        a.append(new)
        summation = new / total
        if i%8==0:
            j = int(i/8)-1
            rise = b[j]/total
            if rise < 0.01 and summation > 0.75 and judge == 0:
                judge = 1
                xaxis = i
                yaxis = new
                if summation>1:
                    sumh = 1
                else:
                    sumh = summation
                print('End of temperature diffusion percentage of insulator is ', end='')
                print(sumh*100, end='')
                print('%')
                if summation > 0.985:
                    print('No defect on the insulator.')
                else:
                    print('Defect exist on the insulator and need further diagnosis.')
    return a, xaxis, yaxis, sumh


y, xa, ya, sumh = f(fault9)
x_major_locator = MultipleLocator(32)  # x坐标
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# fig = plt.figure(graynum_row, y, dpi=128, figsize=(12, 9))
# # 1. fault, c='blue'改颜色
plt.plot(graynum_row, y, c='red', label='Gray scale integral curve', alpha=0.5, linewidth=1.0, linestyle='-', marker='v')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
# 2. good
# plt.plot(graynum_row, y, label='Gray scale integral curve', alpha=0.5, linewidth=1.0, linestyle='-', marker='v')
plt.annotate('Loop ends at %.16f' % sumh, xy=(xa, ya), xytext=(20, 1.1),
             arrowprops=dict(facecolor='black', shrink=0.01), fontsize=17)
plt.fill_between(graynum_row, y, facecolor='blue', alpha=0.1)   # 为两个区域中间填充颜色'red' or 'blue'
plt.savefig('./gray/good/figure9.jpg', dpi=750, bbox_inches='tight')
plt.show()