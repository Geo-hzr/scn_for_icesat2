import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from tensorflow.keras import *
import feature_augmentation
import scene_classification_network

#fix random seed
tf_seed = 0
tf.random.set_seed(tf_seed)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def train_model():

    model = scene_classification_network.scene_classification_net()

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4, decay=1e-8, clipvalue=1.0),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    input_point_cloud, input_normal_vector, input_ad_matrix, \
    input_n_ad_matrix, input_dc_vector, input_image, \
    input_label = feature_augmentation.feature_space_construction()

    index_list = [i for i in range(len(input_label))]
    np.random.shuffle(index_list)
    index_list = index_list[:]# determine how many samples are used

    input_point_cloud = np.array(input_point_cloud)[index_list]
    input_normal_vector = np.array(input_normal_vector)[index_list]
    input_ad_matrix = np.array(input_ad_matrix)[index_list]
    input_n_ad_matrix = np.array(input_n_ad_matrix)[index_list]
    input_dc_vector = np.array(input_dc_vector)[index_list]
    input_image = np.array(input_image)[index_list] / 255.
    input_label = np.array(input_label)[index_list]

    print(input_point_cloud.shape, input_normal_vector.shape, input_ad_matrix.shape, input_dc_vector.shape,
          input_image.shape, input_label.shape)

    y_train_onehot = input_label
    patience_num = 5
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=patience_num, mode='min')  # loss
    validation_split = 0.7
    history = model.fit(
        [input_point_cloud, input_normal_vector, input_ad_matrix, input_dc_vector, input_image, input_n_ad_matrix],
        y_train_onehot
        , epochs=20, batch_size=32, validation_split=validation_split, callbacks=[callback])  #

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

    output_path =r'saved_model/xxx'# determine output path

    model.save(output_path + r'.h5')

train_model()