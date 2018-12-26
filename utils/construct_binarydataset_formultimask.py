# 读取不同类型的数据
import tensorflow as tf
import re
#
from utils.readFileUtil import read_different_type_data
def get_different_types_data(dir, type1, type2, wfire_name):
    # 写文件
    wfile_dir = '../data/'+wfire_name+'.tsv'
    lines = []
    # 读文件
    with open(dir, 'r', encoding='utf-8') as file:
        for line in file:
            results = line.split('\t')
            # print(results)
            if results[1] == type1:
                lines.append(line)
            if results[1] == type2:
                lines.append(line)
        pass
    # 写文件
    with open(wfile_dir, 'w', encoding='utf-8') as wfile:
        for line in lines:
            wfile.write(line)

# 构建5种二类数据集
# true false dataset
get_different_types_data('../data/liar_train.tsv', 'true', 'false', 'liar_true_false')
# true pants-fire dataset
get_different_types_data('../data/liar_train.tsv', 'true', 'pants-fire', 'liar_true_pantsfire')
# true mostly-true dataset
get_different_types_data('../data/liar_train.tsv', 'true', 'mostly-true', 'liar_true_mostlytrue')
# true half-true dataset
get_different_types_data('../data/liar_train.tsv', 'true', 'half-true', 'liar_true_halftrue')
# true barely-true dataset
get_different_types_data('../data/liar_train.tsv', 'true', 'barely-true', 'liar_true_barelytrue')
#
# 9.00数据集构造成功
# 10.00代码成功
# 上传到github