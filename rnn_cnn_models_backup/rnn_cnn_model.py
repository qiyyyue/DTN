#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TRCNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    kg_embedding_dim = 20   # kg dim
    word2vec_dim = 64       #word2vec dim
    seq_length = 3        # 序列长度
    num_classes = 2        # 类别数

    num_filters = 256  # 卷积核数目
    kernel_size = 2  # 卷积核尺寸
    cnn_hidden_dim = 64  # 全连接层神经元

    num_layers= 2           # 隐藏层层数
    rnn_hidden_dim = 64        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 50          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 20      # 每多少轮存入tensorboard


class TripleRCNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.w2v_input_x = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.word2vec_dim], name='w2v_input_x')
        self.kg_input_x = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.kg_embedding_dim], name='kg_input_x')
        self.cnn_input_x = tf.placeholder(tf.float32, [None, None, 64], name='cnn_input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rcnn()

    def rcnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.rnn_hidden_dim)

        def kg_dropout(): # 为每一个rnn核后面加一个dropout层
            cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        def w2v_dropout():
            cell = lstm_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.name_scope("kg_rnn"):
            # 多层rnn网络
            kg_cells = [kg_dropout() for _ in range(self.config.num_layers)]
            kg_rnn_cell = tf.contrib.rnn.MultiRNNCell(kg_cells)

            _kg_outputs, _ = tf.nn.dynamic_rnn(cell=kg_rnn_cell, inputs=self.kg_input_x, dtype=tf.float32)
            kg_last = _kg_outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("kg_score"):
            # 全连接层，后面接dropout以及relu激活
            kg_rnn_fc = tf.layers.dense(kg_last, self.config.rnn_hidden_dim, name='fc_kg')
            kg_rnn_fc = tf.contrib.layers.dropout(kg_rnn_fc, self.keep_prob)
            kg_rnn_fc = tf.nn.relu(kg_rnn_fc)
            self.kg_rnn_fc = kg_rnn_fc

        with tf.name_scope("w2v_rnn"):
            # 多层rnn网络
            w2v_cells = [w2v_dropout() for _ in range(self.config.num_layers)]
            w2v_rnn_cell = tf.contrib.rnn.MultiRNNCell(w2v_cells)

            _w2v_outputs, _ = tf.nn.dynamic_rnn(cell=w2v_rnn_cell, inputs=self.w2v_input_x, dtype=tf.float32)
            w2v_last = _w2v_outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("w2v_score"):
            # 全连接层，后面接dropout以及relu激活
            w2v_rnn_fc = tf.layers.dense(w2v_last, self.config.rnn_hidden_dim, name='fc_w2v')
            w2v_rnn_fc = tf.contrib.layers.dropout(w2v_rnn_fc, self.keep_prob)
            w2v_rnn_fc = tf.nn.relu(w2v_rnn_fc)
            self.w2v_rnn_fc = w2v_rnn_fc

        with tf.name_scope("cnn"):
            # CNN layer
            #print(w2v_rnn_fc)
            #print(tf.to_float(tf.stack([kg_rnn_fc, w2v_rnn_fc], 1)))

            conv = tf.layers.conv1d(tf.to_float(tf.stack([kg_rnn_fc, w2v_rnn_fc], 1)), self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

            cnn_fc = tf.layers.dense(gmp, self.config.cnn_hidden_dim, name='fc1')
            cnn_fc = tf.contrib.layers.dropout(cnn_fc, self.keep_prob)
            cnn_fc = tf.nn.relu(cnn_fc)
            self.cnn_fc = cnn_fc



        with tf.name_scope("detecter"):
            # 分类器
            self.logits = tf.layers.dense(cnn_fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
