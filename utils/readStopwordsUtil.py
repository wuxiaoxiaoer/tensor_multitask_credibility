import os


# 获得中英文停用词表

# 获得中文停用词表
def readCNstopwords(dir):
    file = open(dir, 'r', encoding='utf-8')
    line = file.readline()
    stopwords = []
    while line:
        line = line.strip('\n')
        stopwords.append(line)
        line = file.readline()
        pass
    return stopwords
