import spektral
import tensorflow as tf


class MSDMT(tf.keras.Model):
    def __init__(self,
                 timestep=10,
                 portrait_dim=32,
                 behavior_num=100 + 1,
                 behavior_emb_dim=16,
                 behavior_maxlen=64,
                 behavior_dim=32,
                 network_dim=32,
                 dropout=0.5):
        super(MSDMT, self).__init__()

        self.timestep = timestep
        self.dropout = dropout
        self.portrait_dim = portrait_dim
        self.behavior_num = behavior_num
        self.behavior_emb_dim = behavior_emb_dim
        self.behavior_maxlen = behavior_maxlen
        self.behavior_dim = behavior_dim
        self.network_dim = network_dim

        self.portrait_net = tf.keras.Sequential(
            name='portrait_net',
            layers=[tf.keras.layers.LSTM(units=self.portrait_dim,
                                         return_sequences=False),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Dense(units=self.portrait_dim,
                                          activation='relu',
                                          use_bias=False)])

        self.behavior_net = tf.keras.Sequential(
            name='behavior_net',
            layers=[tf.keras.layers.Embedding(input_dim=self.behavior_num,
                                              output_dim=self.behavior_emb_dim,
                                              mask_zero=True),
                    tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, self.behavior_maxlen, self.behavior_emb_dim))),
                    tf.keras.layers.Conv1D(filters=self.behavior_dim,
                                           kernel_size=3,
                                           padding='same',
                                           activation='relu'),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, self.timestep, self.behavior_dim))),
                    tf.keras.layers.LSTM(units=self.behavior_dim,
                                         return_sequences=False),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Dense(units=self.behavior_dim,
                                          activation='relu',
                                          use_bias=False)])
        self.network_net = tf.keras.Sequential(
            name='network_net',
            layers=[spektral.layers.GCNConv(channels=self.network_dim,
                                            activation='relu'),
                    tf.keras.layers.Dropout(rate=self.dropout),
                    tf.keras.layers.Dense(units=self.network_dim,
                                          activation='relu'))

        self.output1 = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output1')
        self.output2 = tf.keras.layers.Dense(units=1, activation=None, name='output2')

    def call(self, inputs):
        U, B, A = inputs
        H = self.portrait_net(U)
        O = self.behavior_net(B)
        X = tf.keras.layers.Concatenate()([H, O])
        V = self.network_net([X, A])
        output1 = self.output1(V)
        output2 = self.output2(V)
        return output1, output2
