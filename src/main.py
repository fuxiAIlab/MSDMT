import os
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import spektral
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

from model import MSDMT

##############################
seed_value = 2021
lr = 0.0001
epochs = 500
alpha = 0.5
beta = 0.5
timestep = 10
maxlen = 64
##############################


def data_process(timestep=10, maxlen=64):
    df_U = pd.read_csv('./data/sample_data_player_portrait.csv')
    df_B = pd.read_csv('./data/sample_data_behavior_sequence.csv')
    df_G = pd.read_csv('./data/sample_data_social_network.csv')
    df_Y = pd.read_csv('./data/sample_data_label.csv')

    U = df_U.drop(['uid', 'ds'], axis=1).values
    U = U.reshape(-1, timestep, U.shape[-1])
    B = df_B['seq'].apply(lambda x: x.split(',') if pd.notna(x) else []).values
    B = tf.keras.preprocessing.sequence.pad_sequences(sequences=B,
                                                      maxlen=maxlen,
                                                      padding='post')
    B = B.reshape(-1, timestep, maxlen)

    G = nx.from_pandas_edgelist(df=df_G,
                                source='src_uid',
                                target='dst_uid',
                                edge_attr=['weight'])
    A = nx.adjacency_matrix(G)
    A = spektral.layers.GCNConv.preprocess(A).astype('f4')
    y1 = df_Y['churn_label'].values.reshape(-1, 1)
    y2 = np.log(df_Y['payment_label'].values + 1).reshape(-1, 1)

    print('U:', U.shape)
    print('B:', B.shape)
    print('G:', A.shape)
    print('y1:', y1.shape, 'y2:', y2.shape)

    return U, B, A, y1, y2


U, B, A, y1, y2 = data_process(timestep=timestep, maxlen=maxlen)
N = A.shape[0]

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)

for train_index, test_index in kfold.split(U, y1):

    train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=seed_value)

    mask_train = np.zeros(N, dtype=bool)
    mask_val = np.zeros(N, dtype=bool)
    mask_test = np.zeros(N, dtype=bool)
    mask_train[train_index] = True
    mask_val[val_index] = True
    mask_test[test_index] = True

    checkpoint_path = './model/checkpoint-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5,
                                                      mode='min')

    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         mode='auto')

    model = MSDMT(timestep=timestep, behavior_maxlen=maxlen)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={'output_1': tf.keras.losses.BinaryCrossentropy(),
                        'output_2': tf.keras.losses.MeanSquaredError()},
                  loss_weights={'output_1': alpha, 'output_2': beta},
                  metrics={'output_1': tf.keras.metrics.AUC(),
                           'output_2': 'mae'})

    model.fit([U, B, A], [y1, y2],
              validation_data=([U, B, A], [y1, y2], mask_val),
              sample_weight=mask_train,
              batch_size=N,
              epochs=epochs,
              shuffle=False,
              callbacks=[early_stopping, best_checkpoint],
              verbose=1)
