# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
# dropout: 每个元素被保留的概率
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])

# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 28
# 每个隐含层的节点数
hidden_size = 256
# LSTM layer 的层数
layer_num = 1
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])

# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 28, 28])

###########试验田-开始###################
#任务1的LSTM单元
def lstm1():
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

# 公共LSTM
def lstm_common():
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

# tensorflow中只有按行拼接，如果要按列拼接，具体做法是：需要两个矩阵分别先转置，再拼接，再转置。
def concatHiddenAndInput(outputs_single, inputs):
    input_output_conlist = list()
    for timestep in range(timestep_size):
        # 将一个矩阵拼接到另一个矩阵后面
        input_time = inputs[:, timestep, :]
        outputs_single_time = outputs_single[timestep]
        # 转置
        input_trans = tf.transpose(input_time)
        outputs_single_trans = tf.transpose(outputs_single_time)
        # 拼接
        input_output_concat = tf.concat([input_trans, outputs_single_trans], 0)
        # 转置
        input_output_trans = tf.transpose(input_output_concat)
        input_output_conlist.append(input_output_trans)
    return input_output_conlist

# 第一个任务的encoder:
init_state_single = lstm1().zero_state(batch_size, dtype=tf.float32)
outputs_single = list()
state_single = init_state_single
with tf.variable_scope('RNN_single'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态，输入的是x[:,2,:] -->[批大小, token长度] 是(?, 28)
        # 另外，前后的这两个state_single要一致。
        (cell_output, state_single) = lstm1()(X[:, timestep, :], state_single)
        outputs_single.append(cell_output)

# 将上一次的输出outputs_single送入下一次
outputs_common = list()
init_state_common = lstm_common().zero_state(batch_size, dtype=tf.float32)
# 将将隐藏层(?, 256) + 输入层(?, 28)结合起来作为下一层LSTM的输入
new_inputs = concatHiddenAndInput(outputs_single, X)
print("输出第二层输入的值：")
print(new_inputs)
with tf.variable_scope('RNN_common'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态，这里输入的也应该是[批大小, token长度]
        # 要修改outputs_single的值, 将隐藏层(?, 256) + 输入层(?, 28)
        (cell_output, init_state_common) = lstm_common()(new_inputs[timestep], init_state_common)
        outputs_common.append(cell_output)
h_state_common = outputs_common[-1]

# 还得有个for来循环两次‘RNN_single’, 并且是修改成不同的输入。

###########试验田-结束###################

# 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
# 首先定义 softmax 的连接权重矩阵和偏置
# out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
# out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
# 连接两个LSTM的参数节点
y_pre = tf.nn.softmax(tf.matmul(h_state_common, W) + bias)

# 损失和评估函数
# 计算出来的结果和真实结果按位相乘，我觉得是想增大计算值的区分度
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))

train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(5000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1)%200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

# 计算测试数据的准确率
print("test accuracy %g"% sess.run(accuracy, feed_dict={
    _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))