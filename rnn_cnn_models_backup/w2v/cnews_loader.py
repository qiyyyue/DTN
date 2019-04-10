# coding: utf-8

import sys
from collections import Counter
from imp import reload

import numpy as np
import tensorflow.contrib.keras as kr
from gensim.models import Word2Vec

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file_tmp(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('#')
                if content:
                    contents.append(list(content.split("\t")))
                    labels.append(native_content(label))
                else:
                    print(content)
            except:
                print(line)
                pass
    return contents, labels

def read_category():
    """读取分类目录，固定"""
    categories = ['fake', 'true']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def word_to_vec(model_path):
    return Word2Vec.load(model_path)

def list2content(con_lsit):
    content = ""
    for c in con_lsit:
        content += c
    return content

def process_file(filename, word_to_vec, cat_to_id, max_length=30):
    """将文件转换为id表示"""
    contents, labels = read_file_tmp(filename)

    data_id, label_id = [], []
    #print(len(contents))
    for i in range(len(contents)):
        data_id.append([word_to_vec[x] for x in contents[i] if x in word_to_vec])
        label_id.append(cat_to_id[labels[i]])


    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=32):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def file_process_test(filename, word_to_vec, cat_to_id, max_length=3):
    """将文件转换为id表示"""
    contents, labels = read_file_tmp(filename)

    print(len(contents))

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_vec[x] for x in contents[i] if x in word_to_vec])
        label_id.append(cat_to_id[labels[i]])
        #print([word_to_vec[x] for x in contents[i] if x in word_to_vec])
        # if i == 10:
        #     break

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    print(x_pad.shape)
    print(y_pad.shape)
