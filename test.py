# 测试
import tensorflow as tf
from utils.readLiarFile import getliar_text_metadata_vectors
#
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
def test01():
    # [
    # t0 [[0 1 2]]
    # t1 [[3 4 5]]
    # t2 [[6 7 8]]
    # ]
    x = [x for x in range(0, 9)]
    y = [x for x in range(0, 9)]
    x_tf = tf.reshape(x, [-1])
    y_tf = tf.reshape(y, [-1])
    print(x_tf)
    print(y_tf)

    x = tf.reshape(x, [-1, 3, 3])
    x1 = x[-1]
    print(x1)

    xx = [x for x in range(0, 27)]
    xx = tf.reshape(xx, [3, 3, 3])
    # 根据“,”逗号定位到是哪一维，然后再根据数字A定位到这一维的第A一行.
    # 下面这个例子相当于：获得二维数组，取二维数组中的包含着第3行信息。
    x[:, 2, :]

    # 运行
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print(sess.run(x1))
    print("x的值：")
    print(sess.run(x))
    print("输出XX的值：")
    print(xx)
    print(sess.run(xx))
    print("输出XX取部分行的值：")
    print(xx[:, 2, :])
    print(sess.run(xx[:, 2, :]))
    print("输出XX取部分行的值 - 结束")

    print("输出x_tf*y_tf的值：")
    print(sess.run(x_tf * y_tf))

    '''
    x_trans1 = tf.reshape(x, [-1, 3, 3])
    x_trans2 = tf.reshape(x, [3, -1, 3])
    x_trans3 = tf.reshape(x, [3, 3, -1])
    print(x_trans1)
    print("trans 1: ")
    print(sess.run(x_trans1))
    print("trans 2: ")
    print(sess.run(x_trans2))
    print("trans 3: ")
    print(sess.run(x_trans3))
    '''
    pass

# 尝试两个矩阵，将两个矩阵拼接在一起。
def test02():
    matrix1 = [x for x in range(27)]
    matrix2 = [x for x in range(27, 54)]
    mxr1 = tf.reshape(matrix1, [3, 3, 3])
    mxr2 = tf.reshape(matrix2, [3, 3, 3])
    # for i in range(3):
    #     print(sess.run(mxr1[:, i, :]))
    print("输入mxr1中第一行的值：")
    print(sess.run(mxr1[:, 0, :]))
    mxr12 = tf.concat([mxr1[:, 0, :], mxr2[:, 0, :]], 0)
    print("输出拼接结果：")
    print(sess.run(mxr12))
    mxr1tr = tf.transpose(mxr12)
    print("输出转置之后的结果：")
    print(sess.run(mxr1tr))
    print("############第二种方式#########")
    # tensorflow只有按行拼接，没有按列拼接，解决方案是：先转置，再拼接, 再转置。
    # 转置
    mxr10tr = tf.transpose(mxr1[:, 0, :])
    mxr20tr = tf.transpose(mxr2[:, 0, :])
    # 拼接
    mxr1020con = tf.concat([mxr10tr, mxr20tr], 0)
    # 转置
    mxr1020contr = tf.transpose(mxr1020con)
    print("输出凭借之后的结果：")
    print(sess.run(mxr1020contr))

    print("输出整个值：")
    print(sess.run(mxr1))
    print(sess.run(mxr2))
    pass
def test03():
    for i in range(4,3):
        print(i)
# test03()

def test03_get_liar_data():
    start = 0
    length = 128
    # 测试
    article_vec, labels = getliar_text_metadata_vectors(dir='data/liar_train.tsv', start=start, length=length)
    print("文本句子大小--批数量batch_size：")
    print(len(article_vec))
    print("一个句子中的词数量大小--timestep_size：")
    print(len(article_vec[0]))
    print("一个词的嵌入大小--Dimention size：")
    print(len(article_vec[0][0]))
    print("标签：")
    print(labels)
    pass

# 执行方法
test03_get_liar_data()

# 3.00弄好
# 3.30整理好输入输出
# 4.30弄服务器
# 5.30上传到服务器