from tensorflow.keras.layers import *
from tensorflow.keras import  Model
from tensorflow.keras.applications import *
from spektral.layers import *

def build_pointnet(inputs, normal_vec, num_channels):

    # Rotation transformation
    x = Conv1D(1 * num_channels, 1)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv1D(2 * num_channels, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv1D(8 * num_channels, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(4 * num_channels, activation='relu')(x)
    x = Dense(2 * num_channels, activation='relu')(x)
    x = Dense(3 * 3, weights=[np.zeros([2 * num_channels, 3 * 3]), np.eye(3).flatten().astype(np.float32)])(x)
    r_mat = Reshape((3, 3))(x)

    r = Dot(axes=2)([inputs, r_mat])
    r = Conv1D(1 * num_channels, 1)(r)
    r = BatchNormalization()(r)
    r = LeakyReLU(0.01)(r)
    r = Conv1D(1 * num_channels - 3, 1)(r)
    r = BatchNormalization()(r)
    r = LeakyReLU(0.01)(r)

    # Feature transformation
    r = Concatenate()([r, normal_vec])
    x = Conv1D(1 * num_channels, 1)(r)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv1D(2 * num_channels, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv1D(8 * num_channels, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(4 * num_channels, activation='relu')(x)
    x = Dense(2 * num_channels, activation='relu')(x)
    x = Dense((1 * num_channels) * (1 * num_channels), weights=[np.zeros([(2 * num_channels), (1 * num_channels) * (1 * num_channels)]), np.eye((1 * num_channels)).flatten().astype(np.float32)])(x)
    f_mat = Reshape(((1 * num_channels), (1 * num_channels)))(x)

    f = Dot(axes=2)([r, f_mat])
    f = Conv1D(1 * num_channels, 1)(f)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)
    f = Conv1D(2 * num_channels, 1)(f)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)
    f = Conv1D(16 * num_channels, 1)(f)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)

    outputs = GlobalMaxPool1D()(f)

    return outputs

def build_densenet(inputs, num_classes):

    model_path = r'pretrained_model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    dn121 = DenseNet121(
            weights=model_path,
            include_top=False,
            input_tensor=inputs,
            classes=num_classes)
    dn121.trainable = False

    x = dn121(inputs)
    outputs = GlobalAvgPool2D()(x)

    return outputs

def build_gan(inputs, adj_mat, normalized_adj_mat, num_channels, num_heads, num_nodes):

    x = BatchNormalization()(inputs)
    x = GATConv(channels=num_channels, attn_heads=num_heads)([x, adj_mat])
    x = LeakyReLU(0.01)(x)
    x = BatchNormalization()(x)
    x, _ = DMoNPool(k=num_nodes // 2, mlp_hidden=[num_nodes])([x, normalized_adj_mat])
    x = LeakyReLU(0.01)(x)
    outputs = GlobalMaxPool1D()(x)

    return outputs

def build_scn(num_points=2048, num_features=3,
              img_height=128, img_width=512,
              num_nodes=1024, num_channels=1,
              num_classes=2):

    inputs_pcd = Input(shape=(num_points, num_features))

    inputs_normal_vec = Input(shape=(num_points, num_features))

    inputs_img = Input(shape=(img_height, img_width, 3))

    inputs_adj_mat = Input(shape=(num_nodes, num_nodes))

    inputs_normalized_adj_mat = Input(shape=(num_nodes, num_nodes))

    inputs_dc_vec = Input(shape=(num_nodes, num_channels))

    f1 = build_pointnet(inputs_pcd, inputs_normal_vec, num_channels=64)
    f2 = build_densenet(inputs_img, num_classes)
    f3 = build_gan(inputs_dc_vec, inputs_adj_mat, inputs_normalized_adj_mat, num_channels=64, num_heads=16, num_nodes=num_nodes)

    x = Concatenate()([f1, f2, f3])

    weights = Dense(x.shape[-1] // 2, activation='relu')(x)
    weights = Dense(x.shape[-1], activation='sigmoid')(weights)

    x = Multiply()([x, weights])
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs_pcd, inputs_normal_vec, inputs_img, inputs_adj_mat, inputs_normalized_adj_mat, inputs_dc_vec], outputs=outputs)

    print(r'Model initialization done.')

    return model
