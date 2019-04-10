# coding: utf-8

from __future__ import print_function
import tensorflow.contrib.keras as kr
import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from triple2vec.rnn_model_vec import TRNNConfig, TextRNN
from triple2vec.cnews_loader_vec import word_to_vec, read_category, batch_iter, process_file, file_process_test

base_dir = '../data/train_news/2016_election'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')


save_dir = 'checkpoints/vec/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def read_file(file_name):
    re_list = []
    with open(file_name, "r", encoding="utf8") as r_f:
        for line in r_f:
            try:
                line_list = line.strip("/n").split("\t")
                re_list += line_list
            except:
                pass
    return re_list

def file_embedding(file_name):
    word_list = read_file(file_name)
    text_vec = [[w2v[x] for x in word_list if x in w2v]]
    x_pad = kr.preprocessing.sequence.pad_sequences(text_vec, config.seq_length)
    return x_pad


def text2vec(text):
    return

def test_tmp(cat):

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    root_path = "../data/train_news/2016_election/triples/" + cat
    out_path = "../data/train_news/2016_election/vec/" + cat
    files = os.listdir(root_path)
    for file in files:
        try:
            file_name = os.path.join(root_path, file)
            x_pad = file_embedding(file_name)
            if x_pad.shape == (1, 50):
                x_pad = np.zeros((1, 50, 64))
            feed_dict = {
                model.input_x: x_pad,
                model.keep_prob: 1.0
            }
            # print(x_test[start_id:end_id].shape)
            tmp_vec = session.run(model.fc, feed_dict=feed_dict)

            vec_file_name = os.path.join(out_path, file.replace("triple.txt", "vec.txt"))
            np.savetxt(vec_file_name, tmp_vec)
        except Exception as e:
            print(file, e)
            pass

        #print(tmp_vec.shape)
        #break



if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python run_rnn.py [train / test]""")

    print('Configuring RNN model...')
    config = TRNNConfig()
    categories, cat_to_id = read_category()
    model = TextRNN(config)
    w2v = word_to_vec("../model/word2vec_model/brexit/w2v_model.model")

    #file_process_test(train_dir, word_to_vec, cat_to_id, config.seq_length)
    #train()
    test_tmp("fake")
    test_tmp("true")
    # if sys.argv[1] == 'train':
    #     train()
    # else:
    #     test()