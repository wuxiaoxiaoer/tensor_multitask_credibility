# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

############# 记忆加强


# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 首先导入数据，看一下数据的形式
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)

##########################################################################设置模型超参数

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

##########################################################################搭建LSTM模型

# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
####################################################################
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 28, 28])

###########试验田-开始###################
#定义单个LSTM
def lstm2():
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

rnn_cells = []
for iiLyr in range(layer_num):
    rnn_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True))
mlstm_cell_single = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)

init_state_single = mlstm_cell_single.zero_state(batch_size, dtype=tf.float32)
outputs_single = list()
state_single = init_state_single
with tf.variable_scope('RNN_single'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态，输入的是x[:,2,:] -->[批大小, token长度] 是(?, 28)
        # 另外，前后的这两个state_single要一致，为什么？后面慢慢感受。
        (cell_output, state_single) = mlstm_cell_single(X[:, timestep, :], state_single)
        outputs_single.append(cell_output)
    print("outputs_single的值：")
    print(outputs_single)
h_state_single = outputs_single[-1]

# 将上一次的输出outputs_single送入下一次
outputs_second = list()
init_state_second = lstm2().zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_second'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态，这里输入的也应该是[批大小, token长度]
        # ！！！要修改outputs_single的值，并且添加上第一次的输入。state_single要管不？
        print("输出outputs_single的time上的值：")
        print(outputs_single[timestep])
        (cell_output, init_state_second) = lstm2()(outputs_single[timestep], init_state_second)
        outputs_second.append(cell_output)
    print("输出第二次的outputs_single")
    print(outputs_second)
h_state_second = outputs_second[-1]

# 还得有个for来循环两次‘RNN_single’, 并且是修改成不同的输入。

###########试验田-结束###################

stacked_rnn = []
for iiLyr in range(layer_num):
    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True))
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
print("多LSTM输出元组：")
print(mlstm_cell)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# **步骤6：方法二，按时间步展开计算
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
print("输出outputs的值看看：")
print(outputs)
h_state = outputs[-1]

# 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
# 首先定义 softmax 的连接权重矩阵和偏置
# out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
# out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
#y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
#y_pre = tf.nn.softmax(tf.matmul(h_state_single, W) + bias)
y_pre = tf.nn.softmax(tf.matmul(h_state_second, W) + bias)


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