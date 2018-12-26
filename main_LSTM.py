import tensorflow as tf

# 主任务LSTM
class main_lstm(object):
    def __init__(self, hidden_size, timestep_size, batch_size, inputs):
        self.hidden_size = hidden_size
        self.timestep_size = timestep_size
        self.batch_size = batch_size
        self.inputs = inputs
        pass
    # 定义LSTM
    def LSTM(self):
        stacked_rnn = []
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, state_is_tuple=True)
        stacked_rnn.append(lstm_cell)
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
        return mlstm_cell

    def exeRNNByTimes(self):
        # 要添加变量范围，为什么？
        with tf.variable_scope('RNN_commons'):
            init_state_single = self.LSTM().zero_state(self.batch_size, dtype=tf.float32)
            outputs_single = list()
            state_single = init_state_single
            for timestep in range(self.timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                # 这里的state保存了每一层 LSTM 的状态，输入的是x[:,2,:] -->[批大小, token长度] 是(?, 28)
                # 另外，前后的这两个state_single要一致。
                (cell_output, state_single) = self.LSTM()(self.inputs[:, timestep, :], state_single)
                outputs_single.append(cell_output)
            return outputs_single
        pass
    pass