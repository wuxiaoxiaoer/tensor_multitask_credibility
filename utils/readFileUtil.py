# encoding: utf-8
import os
import numpy as np
import re
import linecache

# 读取文件，从start行开始，读取length行文件
# 形式： list[split_list[字段]]
# list[0]:表1行，split_list[0]:表一行中的一个字段
def readFile(dir, start, length):
    list = [];
    curdir = os.getcwd()
    parent_dir = os.path.dirname(curdir)
    file = open(dir, encoding='utf-8')
    i = 0
    k = 0
    while i<length:
        line = linecache.getline(dir, start+k+1)
        k += 1
        mat = re.compile(r'\t')
        split_list = mat.split(line)
        if line:
            # 如果真正能够变为1行，则加1。
            i += 1
            list.append(split_list)
        if not line:
            # start = 0, 当对整个数据集读取完成后，重新从0开始读取，依次循环，无穷尽
            start = 0
    return list

# 继承某个方法：要修改~！
def read_different_type_data(dir, start, length, type):
    list = [];
    curdir = os.getcwd()
    parent_dir = os.path.dirname(curdir)
    file = open(dir, encoding='utf-8')
    i = 0
    k = 0
    while i < length:
        line = linecache.getline(dir, start + k + 1)
        k += 1
        mat = re.compile(r'\t')
        split_list = mat.split(line)
        if line:
            # 如果真正能够变为1行，则加1。
            i += 1
            list.append(split_list)
        if not line:
            # start = 0, 当对整个数据集读取完成后，重新从0开始读取，依次循环，无穷尽
            start = 0
    return list