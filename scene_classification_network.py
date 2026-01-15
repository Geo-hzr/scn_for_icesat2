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

    x = GlobalMaxPool1D()(f)

    return x

def build_densenet(inputs, num_classes):

    model_path = r'pretrained_model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    dn121 = DenseNet121(
        weights=model_path,
        include_top=False,
        input_tensor=inputs,
        classes=num_classes)

    dn121.trainable = False

    x = dn121(inputs)
    x = GlobalAvgPool2D()(x)

    return x

def build_gan(inputs, adj_mat, adj_mat_normalized, num_channels, num_heads, num_nodes):

    x = BatchNormalization()(inputs)
    x = GATConv(channels=num_channels, attn_heads=num_heads)([x, adj_mat])
    x = LeakyReLU(0.01)(x)
    x = BatchNormalization()(x)
    x, _ = DMoNPool(k=num_nodes // 2, mlp_hidden=[num_nodes])([x, adj_mat_normalized])
    x = LeakyReLU(0.01)(x)
    x = GlobalMaxPool1D()(x)

    return x

def build_scn(num_points=2048,
              num_features=3,
              img_height=128,
              img_width=512,
              num_nodes=1024,
              num_channels=1,
              num_classes=2):

    pcd = Input(shape=(num_points, num_features))

    normal_vec = Input(shape=(num_points, num_features))

    img = Input(shape=(img_height, img_width, 3))

    adj_mat = Input(shape=(num_nodes, num_nodes))

    adj_mat_normalized = Input(shape=(num_nodes, num_nodes))

    dc_vec = Input(shape=(num_nodes, num_channels))

    f1 = build_pointnet(pcd, normal_vec, num_channels=64) # 16
    f2 = build_densenet(img, num_classes)
    f3 = build_gan(dc_vec, adj_mat, adj_mat_normalized,
                   num_channels=64, num_heads=16, num_nodes=num_nodes) # 32 8

    x = Concatenate()([f1, f2, f3])

    weights = Dense(x.shape[-1] // 2, activation='relu')(x)
    weights = Dense(x.shape[-1], activation='sigmoid')(weights)

    x = Multiply()([x, weights])
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(
        inputs=[pcd, normal_vec, img, adj_mat, adj_mat_normalized, dc_vec],
        outputs=output)

    print('Model initialization done.')

    return model
