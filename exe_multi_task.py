# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from multi_tasks import lstm1, lstm2, lstm_common, concatHiddenAndInput
# 一、任务1，建立深层LSTM
# 1. 将第二层的LSTM输入改成第一层的输出+第一层的输入(ok)
# 2. 新加一个任务(ok)
# 3. 同样是双层LSTM，但第二层和上一个任务相同(ok)
# 4. 每个任务不同的softmax以及不同的评估(ok)
# 5. 测试(ok)。
# 二、增加Attention机制
# 三、考虑多损失融合策略
# 将不同步骤加入github发布
# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 首先导入数据，看一下数据的形式
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)
# 超参数设置
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
# # dropout: 每个元素被保留的概率
# keep_prob = tf.placeholder(tf.float32, [])
# batch_size = tf.placeholder(tf.int32, [])
# alpha和beta用来平衡两个任务的训练损失
alpha = 0.6
beta = 0.4
# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 28
# 第二个时间节点的长度，其实在本程序中，任务1和任务2,3,...的编译长度应该是一样的，
# 这样才能实现：编译输出后实现时间序列上的拼接作为公共RNN中的输入。
timestep_size_second = 28
# 每个隐含层的节点数
hidden_size = 256
# LSTM layer 的层数
layer_num = 1
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class1_num = 2
class2_num = 2

# _X1 = tf.placeholder(tf.float32, [None, 784])
# _X2 = tf.placeholder(tf.float32, [None, 784])
# y1 = tf.placeholder(tf.float32, [None, class1_num])
# y2 = tf.placeholder(tf.float32, [None, class2_num])


###########试验田-开始###################
def exe_multi_task(X1, X2, y1, y2, keep_prob, batch_size):
    # 下面几个步骤是实现 RNN / LSTM 的关键
    # 第一个任务的encoder:
    init_state_single = lstm1(hidden_size).zero_state(batch_size, dtype=tf.float32)
    outputs_single = list()
    state_single = init_state_single
    with tf.variable_scope('RNN_single'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态，输入的是x[:,2,:] -->[批大小, token长度] 是(?, 28)
            # 另外，前后的这两个state_single要一致。
            (cell_output, state_single) = lstm1(hidden_size)(X1[:, timestep, :], state_single)
            outputs_single.append(cell_output)

    # 第二个任务的encoder:
    init_state_second = lstm2(hidden_size).zero_state(batch_size, dtype=tf.float32)
    outputs_second = list()
    state_second = init_state_second
    with tf.variable_scope('RNN_second'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state_second) = lstm2(hidden_size)(X2[:, timestep, :], state_second)
            outputs_second.append(cell_output)
        pass

    # 将上一次的输出outputs_single送入下一次
    outputs_common = list()
    init_state_common = lstm_common(hidden_size).zero_state(batch_size, dtype=tf.float32)
    # 将将隐藏层(?, 256) + 输入层(?, 28)结合起来作为下一层LSTM的输入
    # 第一个任务的输出+第一个任务的输入+第二个任务的输出+第二个任务的输入，拼接在一起用于公共RNN的训练。
    new_inputs = concatHiddenAndInput(outputs_single, X1, outputs_second, X2, timestep_size)
    print("输出第二层输入的值：")
    print(new_inputs)
    with tf.variable_scope('RNN_common'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态，这里输入的也应该是[批大小, token长度]
            # 要修改outputs_single的值, 将任务1：隐藏层(?, 256) 和 输入层(?, 28) + 任务2：隐藏层(?, 256) 和 输入层(?, 28)
            (cell_output, init_state_common) = lstm_common(hidden_size)(new_inputs[timestep], init_state_common)
            outputs_common.append(cell_output)
    h_state_common = outputs_common[-1]
    # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    # out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
    # out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
    # 开始训练和测试
    # 任务1：
    W = tf.Variable(tf.truncated_normal([hidden_size, class1_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class1_num]), dtype=tf.float32)
    # 连接两个LSTM的参数节点
    y_pre = tf.nn.softmax(tf.matmul(h_state_common, W) + bias)
    # 任务2：
    W2 = tf.Variable(tf.truncated_normal([hidden_size, class2_num], stddev=0.1), dtype=tf.float32)
    bias2 = tf.Variable(tf.constant(0.1, shape=[class2_num]), dtype=tf.float32)
    y_pre2 = tf.nn.softmax(tf.matmul(h_state_common, W2) + bias2)

    # 损失和评估函数
    # 计算出来的结果和真实结果按位相乘，我觉得是想增大计算值的区分度
    cross_entropy1 = -tf.reduce_mean(y1 * tf.log(y_pre))
    cross_entropy2 = -tf.reduce_mean(y2 * tf.log(y_pre2))
    # 最后的训练集合点
    train_op = tf.train.AdamOptimizer(lr).minimize(alpha * cross_entropy1 + beta * cross_entropy2)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    correct_prediction2 = tf.equal(tf.argmax(y_pre2, 1), tf.argmax(y2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # return train_op, accuracy1, accuracy2
    return h_state_common, cross_entropy1, cross_entropy2, accuracy1, accuracy2
###########试验田-结束###################

#
# # 下面是验证这个多任务的执行代码
# train_op, accuracy1, accuracy2 = exe_multi_task()
#
# sess.run(tf.global_variables_initializer())
# for i in range(5000):
#     _batch_size = 128
#     batch1 = mnist.train.next_batch(_batch_size)
#     # batch2 = mnist.train.next_batch(_batch_size)
#     if (i+1)%200 == 0:
#         train_accuracy, train_accuracy2 = sess.run([accuracy1, accuracy2], feed_dict={
#             _X1:batch1[0], y1: batch1[1], _X2:batch1[0], y2: batch1[1], keep_prob: 1.0, batch_size: _batch_size})
#         # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
#         print("Iter%d, step %d, training accuracy1 %g, training accuracy2 %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy, train_accuracy2))
#     sess.run(train_op, feed_dict={_X1: batch1[0], y1: batch1[1], _X2:batch1[0], y2: batch1[1], keep_prob: 0.5, batch_size: _batch_size})
#
# # 计算测试数据的准确率
# print("test accuracy1 %g, accuracy2 %g"% sess.run(accuracy1, accuracy2, feed_dict={
#     _X1: mnist.test.images, y1: mnist.test.labels, _X2:batch1[0], y2: batch1[1], keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))