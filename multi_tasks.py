import tensorflow as tf
# 多任务组件：lstm1,2,3,4以及公共lstm, 还有拼接各个任务的输出(output_-1)和输入(inputs)
def lstm1(hidden_size):
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

# 任务2的LSTM单元
def lstm2(hidden_size):
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

# 公共LSTM
def lstm_common(hidden_size):
    stacked_rnn = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    stacked_rnn.append(lstm_cell)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    return mlstm_cell

# tensorflow中只有按行拼接，如果要按列拼接，具体做法是：需要两个矩阵分别先转置，再拼接，再转置。
def concatHiddenAndInput(outputs_single, inputs1, outputs_second, inputs2, timestep_size):
    input_output_conlist = list()
    # 在这里尽可能的保证各个任务的输入时间序列相同。如果不同，时间序列长的数据多出来的部分将丢失。
    for timestep in range(timestep_size):
        # 将一个矩阵拼接到另一个矩阵后面
        # 任务1的输出和输入进行拼接
        input_time = inputs1[:, timestep, :]
        outputs_single_time = outputs_single[timestep]
        # 转置
        input_trans = tf.transpose(input_time)
        outputs_single_trans = tf.transpose(outputs_single_time)
        # 拼接
        input_output_concat = tf.concat([input_trans, outputs_single_trans], 0)
        # 转置
        input_output_trans = tf.transpose(input_output_concat)
        input_output_conlist.append(input_output_trans)
        # 任务2的输出和输入进行拼接
        input2_time = inputs2[:, timestep, :]
        outputs_second_time = outputs_second[timestep]
        input2_trans = tf.transpose(input2_time)
        outputs_second_trans = tf.transpose(outputs_second_time)
        input2_output2_concat = tf.concat([input2_trans, outputs_second_trans], 0)
        input2_output2_trans = tf.transpose(input2_output2_concat)
        # 将任务2的输出和输入拼接进去
        input_output_conlist.append(input2_output2_trans)
    return input_output_conlist