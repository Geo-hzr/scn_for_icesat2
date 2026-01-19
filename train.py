import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from tensorflow.keras import *
import feature_augmentation
import scene_classification_network

TF_SEED = 0
tf.random.set_seed(TF_SEED)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

PATIENCE = 3
VALIDATION_SPLIT = 0.7

def train_model():

    model = scene_classification_network.build_scn()

    model.summary()

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4, decay=1e-8, clipvalue=1.0),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, \
    adj_mat_normalized_lst, dc_vec_lst, \
    label_lst = feature_augmentation.construct_feature_space()

    idx_lst = [i for i in range(len(label_lst))]
    np.random.shuffle(idx_lst)

    pcd_lst = np.array(pcd_lst)[idx_lst]
    normal_vec_lst = np.array(normal_vec_lst)[idx_lst]
    img_lst = np.array(img_lst)[idx_lst] / 255.
    adj_mat_lst = np.array(adj_mat_lst)[idx_lst]
    adj_mat_normalized_lst = np.array(adj_mat_normalized_lst)[idx_lst]
    dc_vec_lst = np.array(dc_vec_lst)[idx_lst]
    label_lst = np.array(label_lst)[idx_lst]

    print(pcd_lst.shape, normal_vec_lst.shape, img_lst.shape, adj_mat_lst.shape, dc_vec_lst.shape, label_lst.shape)

    callback = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')

    history = model.fit(
        [pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, adj_mat_normalized_lst, dc_vec_lst],
        label_lst, epochs=20, batch_size=32, validation_split=VALIDATION_SPLIT, callbacks=[callback])

    print(history.history.keys())

    acc = history.history['binary_accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.subplot(1, 2, 1)
    plt.scatter(epochs, acc, label='Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(color='black', linestyle='--')
    plt.subplot(1, 2, 2)
    plt.scatter(epochs, loss, label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(color='black', linestyle='--')
    plt.show()

    output_path =r'saved_model/scn'

    model.save(output_path + r'.h5')

train_model()
