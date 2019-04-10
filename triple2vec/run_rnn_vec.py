# coding: utf-8

from __future__ import print_function

import random

import tensorflow.contrib.keras as kr
import os
import sys
import re
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from numpy import mat
from sklearn import metrics

from att.attention_model import att_model
from triple2vec.rnn_model_vec import TRNNConfig, TextRNN
from triple2vec.cnews_loader_vec import word_to_vec, read_category, batch_iter, process_file, file_process_test
from model.kg_embedding_model.triple2vec import triple2vec

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


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/vec/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, w2v, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, w2v, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            # if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                # print("No optimization for a long time, auto-stopping...")
                # flag = True
                # break  # 跳出循环
        if flag:  # 同上
            break


def test_tmp():

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    with open("../data/train_news/2016_election/cnews.train.txt", "r", encoding="utf8") as train_f:
        i = 1
        for line in train_f:
            text_str = ""
            text = ""
            try:
                label, content = line.strip().split('#')
                if content:
                    text_str = content
                    text = list(content.split("\t"))

                    #print(text)
                    text_vec = [[w2v[x] for x in text if x in w2v]]
                    #print("vec", text_vec)
                    x_pad = kr.preprocessing.sequence.pad_sequences(text_vec, config.seq_length)

                    feed_dict = {
                        model.input_x: x_pad,
                        model.keep_prob: 1.0
                    }
                    # print(x_test[start_id:end_id].shape)
                    tmp_vec = session.run(model.fc, feed_dict=feed_dict)

                    print(tmp_vec)
                    # tmp_file_name = "data/temp_data/vec/val/vec_" + str(i) + ".txt"
                    # np.savetxt(tmp_file_name, tmp_vec)

            except Exception as e:
                print("erro", e)
                pass

            if i == 2:
                break
            i += 1



def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, w2v, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def is_contain_word(sent, word):
    for w in word.split(" "):
        if w in sent:
            return True
    return False

def find_sent_by_triple(triple, file_name):
    words = list(triple.split("\t"))
    file_name = file_name.replace("_triple.txt", ".txt").replace("triples/true", "news/true_news")
    #print("filename", file_name)
    with open(file_name, "r", encoding="utf8") as rf:
        text = rf.read()
        sents = text.split(".")
        #print("sents", sents)
        #print("words", words)
        for sent in sents:
            tag = True
            for word in words:
                if not is_contain_word(sent, word):
                    tag = False
            if tag:
                return sent
    return ""

def read_news(filename):
    with open(filename, "r", encoding="utf8") as f:
        return f.read()

def cal_bias(label):
    fake_base_dir = "../data/train_news/2016_election/triples/" + label
    t2v = triple2vec(20)
    i = 1
    for file in os.listdir(fake_base_dir):
        file_name = os.path.join(fake_base_dir, file)
        tmp_rf = open(file_name, "r", encoding="utf8")

        count = 0
        sum_bias = 0.0
        max_bias = 0.0
        avg_bias = 0.0

        for line in tmp_rf:
            triple = list(line.split("\t"))
            h_mat = mat(t2v.entity2vec(triple[0]))
            r_mat = mat(t2v.relation2vec(triple[1]))
            t_mat = mat(t2v.entity2vec(triple[2]))
            if str(h_mat.tolist()).find('None') != -1 or str(t_mat.tolist()).find('None') != -1 or str(
                    r_mat.tolist()).find('None') != -1:
                continue

            tmp_mat = h_mat + r_mat - t_mat
            tmp_bias = 0.0
            for j in range(tmp_mat.shape[1]):
                tmp_bias += np.square(tmp_mat[0, j])

            tmp_bias = np.sqrt(tmp_bias)
            sum_bias += tmp_bias
            if (tmp_bias > max_bias):
                max_bias = tmp_bias
            count += 1
        if count != 0:
            avg_bias = sum_bias/count

        tmp_wf = open("bias_result/" + file.replace("triple", "bias"), "w", encoding="utf8")
        tmp_wf.write(str(max_bias) + "\t" + str(avg_bias))
        tmp_wf.close()
        tmp_rf.close()

        print(str(i))
        i += 1



def triple_iter(label):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    base_dir = "../data/train_news/2016_election/triples/" + label
    i = 1
    for file in os.listdir(base_dir):
        file_name = os.path.join(base_dir, file)

        att = att_model(w2v=w2v, file_name=file_name)
        att.build()

        with open(file_name, "r", encoding="utf8") as rf:
            text_str = ""
            text = ""
            j = 1
            wf = open("result/" + file.replace("triple", "res"), "w", encoding="utf8")
            wf.write(read_news(file_name.replace("_triple.txt", ".txt").replace("triples/" + label, "news/" + label + "_news")) + "\n")
            wf.write("#########################" + "\n")
            for line in rf:
                try:
                    #text = text.strip("\n")
                    text = list(line.split("\t"))
                    text_vec = [[w2v[x] for x in text if x in w2v]]

                    tmp_att_weight = att.cal_triple_att(text_vec[0])
                    #print(tmp_att_weight)

                    x_pad = kr.preprocessing.sequence.pad_sequences(text_vec, config.seq_length)
                    # if x_pad.shape == (1, 50):
                    #     x_pad = np.zeros((1, 50, 64))
                    feed_dict = {
                        model.input_x: x_pad,
                        model.keep_prob: 1.0
                    }
                    # print(x_test[start_id:end_id].shape)
                    tmp_vec = session.run(model.logits, feed_dict=feed_dict)
                    y_pred = session.run(tf.argmax(tf.nn.softmax(tmp_vec), 1))



                    # print("vev", tmp_vec)
                    # print("pred", y_pred)
                    # if y_pred[0] == 1:

                    rattio = np.math.fabs(tmp_vec[0][0] + tmp_vec[0][1])
                    #print("rattio: " + str(rattio))
                    tmp_att_weight = tmp_att_weight*(rattio*5 + random.uniform(0.9, 1.1))
                    if tmp_att_weight > 1:
                        tmp_att_weight = 1

                    added_conf = 1
                    if y_pred[0] == 0:
                        added_conf = 1
                    else:
                        added_conf = 0

                    if True:
                        sent = find_sent_by_triple(line.strip("\n"), file_name)
                        if sent != "":
                            # print("filename:", file_name)
                            # print("vec:", tmp_vec)
                            # print("pred:", y_pred)
                            # print("sent:", sent)
                            # print("triple:", text)
                            # print("conf:", str(tmp_vec[0][0] + tmp_vec[0][1]))
                            # print("--------------")

                            w_str = "pred: " + str(y_pred[0]) + "\n"\
                                    + "sent: " + sent + "\n"\
                                    + "triple: " + line\
                                    + "conf: " + str(tmp_vec[0][0]) + "\t" + str(tmp_vec[0][1]) + "\n"\
                                    + "att_weight: " + str(tmp_att_weight) + "\n" \
                                    + "added conf:" + str(added_conf) + "\n" \
                                    + "------------------------" + "\n"
                            # print(w_str)
                            # print("##########################")
                            # with open("out/" + file.replace("triple", "res_" + str(j)), "w", encoding="utf8") as f:
                            wf.write(w_str)
                            j += 1
                        else:
                            # if sent != "":
                            # print("filename:", file_name)
                            # print("vec:", tmp_vec)
                            # print("pred:", y_pred)
                            # print("sent:", sent)
                            # print("triple:", text)
                            # print("conf:", str(tmp_vec[0][0] + tmp_vec[0][1]))
                            # print("--------------")

                            w_str = "pred: " + str(y_pred[0]) + "\n" \
                                    + "sent: " + "didn't find!" + "\n" \
                                    + "triple: " + line \
                                    + "conf: " + str(tmp_vec[0][0]) + "\t" + str(tmp_vec[0][1]) + "\n" \
                                    + "att_weight: " + str(tmp_att_weight) + "\n" \
                                    + "added conf:" + str(added_conf) + "\n" \
                                    + "------------------------" + "\n"
                            # print(w_str)
                            # print("##########################")
                            # with open("out/" + file.replace("triple", "res_" + str(j)), "w", encoding="utf8") as f:
                            wf.write(w_str)
                            j += 1
                    # print("--------------")
                except Exception as E:
                    print(E)
                    #print(line)
                    pass
            wf.close()
            rf.close()
        # if i == 10:
        #     break
        print(str(i))
        i += 1

def find_sent():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    fake_base_dir = "../data/train_news/2016_election/triples/true"
    i = 1
    for file in os.listdir(fake_base_dir):
        file_name = os.path.join(fake_base_dir, file)

        att = att_model(w2v=w2v, file_name=file_name)
        att.build()

        with open(file_name, "r", encoding="utf8") as rf:
            text_str = ""
            text = ""
            j = 1
            wf = open("result/" + file.replace("triple", "res"), "w", encoding="utf8")
            wf.write(read_news(file_name.replace("_triple.txt", ".txt").replace("triples/true", "news/true_news")) + "\n")
            wf.write("#########################" + "\n")
            for line in rf:
                try:
                    #text = text.strip("\n")
                    text = list(line.split("\t"))
                    text_vec = [[w2v[x] for x in text if x in w2v]]

                    tmp_att_weight = att.cal_triple_att(text_vec[0])
                    #print(tmp_att_weight)

                    x_pad = kr.preprocessing.sequence.pad_sequences(text_vec, config.seq_length)
                    # if x_pad.shape == (1, 50):
                    #     x_pad = np.zeros((1, 50, 64))
                    feed_dict = {
                        model.input_x: x_pad,
                        model.keep_prob: 1.0
                    }
                    # print(x_test[start_id:end_id].shape)
                    tmp_vec = session.run(model.logits, feed_dict=feed_dict)
                    y_pred = session.run(tf.argmax(tf.nn.softmax(tmp_vec), 1))



                    # print("vev", tmp_vec)
                    # print("pred", y_pred)
                    # if y_pred[0] == 1:

                    rattio = np.math.fabs(tmp_vec[0][0] + tmp_vec[0][1])
                    #print("rattio: " + str(rattio))
                    tmp_att_weight = tmp_att_weight*(rattio*5 + random.uniform(0.9, 1.1))
                    if tmp_att_weight > 1:
                        tmp_att_weight = 1

                    added_conf = 1
                    if y_pred[0] == 0:
                        added_conf = 1
                    else:
                        added_conf = 0

                    if True:
                        sent = find_sent_by_triple(line.strip("\n"), file_name)
                        if sent != "":
                            # print("filename:", file_name)
                            # print("vec:", tmp_vec)
                            # print("pred:", y_pred)
                            # print("sent:", sent)
                            # print("triple:", text)
                            # print("conf:", str(tmp_vec[0][0] + tmp_vec[0][1]))
                            # print("--------------")

                            w_str = "pred: " + str(y_pred[0]) + "\n"\
                                    + "sent: " + sent + "\n"\
                                    + "triple: " + line\
                                    + "conf: " + str(tmp_vec[0][0]) + "\t" + str(tmp_vec[0][1]) + "\n"\
                                    + "att_weight: " + str(tmp_att_weight) + "\n" \
                                    + "added conf:" + str(added_conf) + "\n" \
                                    + "------------------------" + "\n"
                            # print(w_str)
                            # print("##########################")
                            # with open("out/" + file.replace("triple", "res_" + str(j)), "w", encoding="utf8") as f:
                            wf.write(w_str)
                            j += 1
                        else:
                            # if sent != "":
                            # print("filename:", file_name)
                            # print("vec:", tmp_vec)
                            # print("pred:", y_pred)
                            # print("sent:", sent)
                            # print("triple:", text)
                            # print("conf:", str(tmp_vec[0][0] + tmp_vec[0][1]))
                            # print("--------------")

                            w_str = "pred: " + str(y_pred[0]) + "\n" \
                                    + "sent: " + "didn't find!" + "\n" \
                                    + "triple: " + line \
                                    + "conf: " + str(tmp_vec[0][0]) + "\t" + str(tmp_vec[0][1]) + "\n" \
                                    + "att_weight: " + str(tmp_att_weight) + "\n" \
                                    + "added conf:" + str(added_conf) + "\n" \
                                    + "------------------------" + "\n"
                            # print(w_str)
                            # print("##########################")
                            # with open("out/" + file.replace("triple", "res_" + str(j)), "w", encoding="utf8") as f:
                            wf.write(w_str)
                            j += 1
                    # print("--------------")
                except Exception as E:
                    print(E)
                    #print(line)
                    pass
            wf.close()
            rf.close()
        # if i == 10:
        #     break
        print(str(i))
        i += 1

def modi_filename():
    tmp_base_dir = 'triples_select_results'
    for file in os.listdir(tmp_base_dir):
        file_name = os.path.join(tmp_base_dir, file)
        if file_name.__contains__('ake'):
            new_name = file_name.replace('ake', 'fake')
            os.rename(file_name, new_name)
        elif file_name.__contains__('true'):
            continue
        else:
            new_name = file_name.replace('rue', 'true')
            os.rename(file_name, new_name)


if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python run_rnn.py [train / test]""")

    print('Configuring RNN model...')
    config = TRNNConfig()
    categories, cat_to_id = read_category()
    model = TextRNN(config)
    w2v = word_to_vec("../model/word2vec_model/2016_election/w2v_model.model")


    cal_bias("fake")
    #find_sent()
    #file_process_test(train_dir, word_to_vec, cat_to_id, config.seq_length)
    #train()
    #test_tmp()
    # test()
    # if sys.argv[1] == 'train':
    #     train()
    # else:
    #     test()
