import tensorflow as tf
from main_LSTM import main_lstm
from attention import attention
def main_task(query, X, hidden_size, timestep_size, batch_size):

    # LSTM
    lstm = main_lstm(hidden_size, timestep_size, batch_size, X)
    outputs_lstm = lstm.exeRNNByTimes()
    # 将张量列表转换为张量：
    out_convert = tf.convert_to_tensor(outputs_lstm)
    # 将张量的第一维和第二维互换
    out_trans = tf.transpose(out_convert, [1, 0, 2])
    # 注意力机制, query
    output_attention = attention(out_trans, query, hidden_size)
    print("output_attention:")
    print(output_attention)
    h_state_main = output_attention
    # h_state_common = outputs_lstm[-1]

    return h_state_main