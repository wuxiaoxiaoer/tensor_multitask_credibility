import tensorflow as tf

# Attention mechanism
# 在Attention中, 输入的是两部分：一个是词向量query，另一个是词序列隐藏层的输出序列（或称为源序列）keys(按照时间步骤)
# 具体计算步骤：
# 1. 加入两个输入
# 2. 计算词向量和隐藏层输出序列中每个步骤key_i之间的相似性(得分函数 score function)：计算方式包括：
#    (1). 点积：similarity_i(query, key_i) = query * key_i
#    (2). cosine相似性: similarity_i(query, key_i) = (query*key_i)/(||query||*||key_i||)
#    (3). MLP网络: similarity_i(query, key_i) = MLP(query, key_i)
# 3. 利用softmax对第2步骤的similarity_i得分求归一化系数alpha_i
#     alpha_i = softmax(sim_i) = exp(simi_i)/sum_j(exp(sim_j))
# 4. 其中步骤3中的结果即为输出序列key_i的权重系数，对输出序列中所有key进行加权求和即可得到attention数值：
#    Attention(query, values(若keys和values是相同的，则为keys)) = sum_i^L(alpha_i * value_i(或key_i))
# 目前绝大多数具体的注意力机制计算方法都符合以上的计算步骤。

def attention(inputs, query, hidden_size):
    query_size_list = query.get_shape().as_list()
    # 获得query的长度大小，这里和hidden_size是相同的
    attention_size = query_size_list[0]

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    # u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    with tf.name_scope('inputs_mlp'):
        # inputs_mlp = (B, T, D)*(D, A) = (B, T, A)
        inputs_mlp = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        pass
    # (B, T, A)*(A, 1) = (B,T)
    inputs_mlp_query = tf.tensordot(inputs_mlp, query, axes=1, name='inputs_mlp_query')
    # 3. 得到系数alphas: (B,T)
    alphas = tf.nn.softmax(inputs_mlp_query, name='alphas')
    # 输出的是(B,D)
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    print("输出attention后的outputs: ")
    print(output)
    return output