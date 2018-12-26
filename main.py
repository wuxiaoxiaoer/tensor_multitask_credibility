import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from main_task import main_task
from exe_multi_task import exe_multi_task
from utils.readLiarFile import getliar_text_metadata_vectors
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)
# 超参数设置
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
# dropout: 每个元素被保留的概率
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])
# alpha和beta用来平衡两个任务的训练损失
alpha = 0.6
beta = 0.4
# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 200
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 50
# 第二个时间节点的长度，其实在本程序中，任务1和任务2,3,...的编译长度应该是一样的，
# 这样才能实现：编译输出后实现时间序列上的拼接作为公共RNN中的输入。
timestep_size_second = 50
# 每个隐含层的节点数
hidden_size = 256
# LSTM layer 的层数
layer_num = 1
# 最后输出分类类别数量，如果是回归预测的话应该是 1
# 主任务标签
class_num = 6
# 多任务标签
class1_num = 2
class2_num = 2
#主任务的输入
# _X = tf.placeholder(tf.float32, [None, 784])
_X = tf.placeholder(tf.float32, [None, 50, 200])
y = tf.placeholder(tf.float32, [None, class_num])

#多任务的输入
_X1 = tf.placeholder(tf.float32, [None, 50, 200])
_X2 = tf.placeholder(tf.float32, [None, 50, 200])
# _X1 = tf.placeholder(tf.float32, [None, 784])
# _X2 = tf.placeholder(tf.float32, [None, 784])
y1 = tf.placeholder(tf.float32, [None, class1_num])
y2 = tf.placeholder(tf.float32, [None, class2_num])

# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 50, 200])
X1 = tf.reshape(_X1, [-1, 50, 200])
X2 = tf.reshape(_X2, [-1, 50, 200])
# X2 = tf.reshape(_X2, [-1, 28, 28])
# 多任务
h_state_common, cross_entropy1, cross_entropy2, accuracy1, accuracy2 = exe_multi_task(X1, X2, y1, y2, keep_prob, batch_size)
# 对tensor按列求均值
query = tf.reduce_mean(h_state_common, axis=0)
# query = [0.1, 0.2, 0.3, 0.4, 0.02, 0.5, 0.08, 0.2]
query = tf.convert_to_tensor(query)
# 主任务
h_state_main = main_task(query, X, hidden_size, timestep_size, batch_size)

# 构造softmax
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state_main, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))

# train_op = tf.train.AdamOptimizer(lr).minimize(0.4*cross_entropy + 0.3*cross_entropy1 + 0.3*cross_entropy2)
train_op = tf.train.AdamOptimizer(lr).minimize(0.4*cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

for i in range(500):
    _batch_size = 32
    # batch1 = mnist.train.next_batch(_batch_size)
    # batch2 = mnist.train.next_batch(_batch_size)
    # 主任务数据的输入：
    article_vec, labels = getliar_text_metadata_vectors(dir='data/liar_train.tsv', start=i*_batch_size, length=_batch_size, is_maintask=True)
    # 多任务数据的输入：
    article_multitask1, labels_multitask1 = getliar_text_metadata_vectors(dir='data/liar_true_false.tsv', start=i*_batch_size, length=_batch_size, is_maintask=False)
    article_multitask2, labels_multitask2 = getliar_text_metadata_vectors(dir='data/liar_true_halftrue.tsv', start=i*_batch_size, length=_batch_size, is_maintask=False)
    if (i+1)%5 == 0:
        # train_accuracy = sess.run(accuracy, feed_dict={
        #     _X: article_vec, y: labels, keep_prob: 1.0,batch_size: _batch_size})
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:article_vec, y: labels, _X1: article_multitask1,  y1: labels_multitask1, _X2: article_multitask2, y2: labels_multitask2, keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: article_vec, y: labels, _X1: article_multitask1, y1: labels_multitask1, _X2: article_multitask2, y2: labels_multitask2, keep_prob: 0.5, batch_size: _batch_size})
    # sess.run(train_op,
    #          feed_dict={_X: article_vec, y: labels, keep_prob: 0.5, batch_size: _batch_size})

# 测试数据集的准确率
for i in range(10):
    article_vec_test, labels_test = getliar_text_metadata_vectors(dir='data/liar_test.tsv', start=i*_batch_size, length=_batch_size, is_maintask=True)
    print("test accuracy %d, %g"% (i, sess.run(accuracy, feed_dict={
        _X: article_vec_test, y: labels_test, _X1: article_multitask1, y1: labels_multitask1, _X2: article_multitask2, y2: labels_multitask2,
        keep_prob: 1.0, batch_size: _batch_size})))

# # 计算测试数据的准确率
# print("test accuracy %g"% sess.run(accuracy, feed_dict={
#     _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))



# 建立五种二分类多任务, (true, barely-true), (true, false), (true, half-true), (true, mostly-true), (true, pants-on-fire)
# 在read_different_types文件中实现
# 首先获得数据集这五种数据集数据（六分类数据）
# 修改多分类结构
# 构造完成，训练

# 看多损失融合的文章

# 画图，看论文确定要画的相关的图