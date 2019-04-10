#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

import os

def _read_file(filename):
    """读取一个文件并转换为一行"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            re = f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')
            return re
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='gbk') as f:
            return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def _read_file_tmp(filename):
    """读取一个文件并转换为一行"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            re = f.read()
            return re
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='gbk') as f:
            return f.read()

def content_deal(content):
    re_content = ""
    for triple in content.split("\n"):
        re_content += triple.replace("\n", " ").replace("\t", " ") + " . "
    return re_content

def save_file(dirname):
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    f_train = open('../data/cnews/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('../data/cnews/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('../data/cnews/cnews.val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):   # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file_tmp(filename)
            if content_deal(content) == " . ":
                continue
            if count < 800:
                f_train.write(category + '\t' + content_deal(content) + '\n')
            elif count < 1000:
                f_test.write(category + '\t' + content_deal(content) + '\n')
            elif count < 1200:
                f_val.write(category + '\t' + content_deal(content) + '\n')
            else:
                break
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == '__main__':
    save_file('oringin_news_data/brexit_news/triples')
    print(len(open('../data/cnews/cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('../data/cnews/cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('../data/cnews/cnews.val.txt', 'r', encoding='utf-8').readlines()))
