import tensorflow as tf


class MSDMT(tf.keras.Model):
    def __init__(self,
                 portrait_input_dim=32,
                 portrait_output_dim=64,

                 behavior_num=100,
                 behavior_emb_dim=64,
                 behavior_input_dim=64,
                 behavior_output_dim=64,

                 lstm_dim=32,
                 dense_dim=32,
                 head_dim=16,
                 embedding_dim=32,
                 item_input_dim=3972 + 1,
                 item_output_dim=32,
                 behavior_input_dim=4 + 1,
                 behavior_output_dim=4,
                 dropout=0.5,
                 **kwargs):
        self.portrait_input_dim = portrait_input_dim
        self.portrait_output_dim = portrait_output_dim

        self.behavior_num = behavior_num
        self.behavior_emb_dim = behavior_emb_dim
        self.behavior_input_dim = behavior_input_dim
        self.behavior_output_dim = behavior_output_dim

        self.lstm = tf.keras.layers.LSTM(units=self.portrait_output_dim, return_sequences=False)

        self.embedding = tf.keras.layers.Embedding(input_dim=self.behavior_num, output_dim=self.behavior_output_dim,
                                                   mask_zero=True)
        self.conv = tf.keras.layers.Conv1D(filters=self.behavior_output_dim, kernel_size=3, padding='same',
                                           activation='relu')
        self.lstm = tf.keras.layers.LSTM(units=self.behavior_output_dim, return_sequences=False)

        self.gcn =

    def call(self, inputs):
        seq1_input1, seq1_input2, seq2_input1, seq2_input2, seq3_input1, seq3_input2 = inputs
