from tensorflow.keras.layers import *
from tensorflow.keras import  Model
from tensorflow.keras.applications import *
from spektral.layers import *

def pretrained_dense_net(inputs, num_classes):

    x = inputs

    model_path = r'pretrained_model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    DN = DenseNet121(
        weights=model_path,
        include_top=False,
        input_tensor=x,
        classes=num_classes)

    DN.trainable = False#True
    x = DN(x)
    x = GlobalAvgPool2D()(x)

    return x

def graph_attention_net(input_x, input_a, input_a_n, channels, attn_heads, num_nodes):

    x = BatchNormalization()(input_x)
    x = GATConv(channels=channels, attn_heads=attn_heads)([x, input_a])
    x = LeakyReLU(0.01)(x)
    x = BatchNormalization()(x)
    x, a = DMoNPool(k=num_nodes // 2, mlp_hidden=[num_nodes])([x, input_a_n])
    x = LeakyReLU(0.01)(x)
    x = GlobalMaxPool1D()(x)

    return x

def modified_point_net(input_data, normal_vector, channels):

    # rotation transformation
    x = Conv1D(1 * channels, 1)(input_data)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv1D(2 * channels, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv1D(8 * channels, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(4 * channels, activation='relu')(x)
    x = Dense(2 * channels, activation='relu')(x)
    x = Dense(3 * 3, weights=[np.zeros([2 * channels, 3 * 3])
        , np.eye(3).flatten().astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    g = Dot(axes=2)([input_data, input_T])
    g = Conv1D(1 * channels, 1)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(0.01)(g)
    g = Conv1D(1 * channels - 3, 1)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(0.01)(g)

    # feature transformation
    g = Concatenate()([g, normal_vector])
    f = Conv1D(1 * channels, 1)(g)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)
    f = Conv1D(2 * channels, 1)(f)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)
    f = Conv1D(8 * channels, 1)(f)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.01)(f)
    f = GlobalMaxPool1D()(f)
    f = Dense(4 * channels, activation='relu')(f)
    f = Dense(2 * channels, activation='relu')(f)
    f = Dense((1 * channels) * (1 * channels), weights=[np.zeros([(2 * channels), (1 * channels) * (1 * channels)])
            , np.eye((1 * channels)).flatten().astype(np.float32)])(f)
    feature_T = Reshape(((1 * channels), (1 * channels)))(f)

    g = Dot(axes=2)([g, feature_T])
    g = Conv1D(1 * channels, 1)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(0.01)(g)
    g = Conv1D(2 * channels, 1)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(0.01)(g)
    g = Conv1D(16 * channels, 1)(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(0.01)(g)

    global_feature = GlobalMaxPool1D()(g)

    return global_feature

def scene_classification_net(num_points=2048,
                             num_features=3,
                             img_width=512,
                             img_height=128,
                             num_nodes=1024,
                             num_channels=1,
                             num_classes=2):

    input_3d_point_cloud = Input(shape=(num_points, num_features))

    input_normal_vector = Input(shape=(num_points, num_features))

    input_ad_matrix = Input(shape=(num_nodes, num_nodes))

    input_n_ad_matrix = Input(shape=(num_nodes, num_nodes))

    input_dc_vector = Input(shape=(num_nodes, num_channels))

    input_2d_image = Input(shape=(img_height, img_width, 3))

    h1 = modified_point_net(input_3d_point_cloud, input_normal_vector, channels=16)
    h2 = pretrained_dense_net(input_2d_image, num_classes)
    h3 = graph_attention_net(input_dc_vector, input_ad_matrix, input_n_ad_matrix,
                                   channels=32, attn_heads=8,num_nodes=num_nodes)

    x = Concatenate()([h1, h2, h3])

    channel_num = x.shape[-1]

    weight = Dense(channel_num // 2, activation='relu')(x)
    weight = Dense(channel_num, activation='sigmoid')(weight)

    x = Multiply()([x, weight])
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(
        inputs=[input_3d_point_cloud, input_normal_vector, input_ad_matrix, input_dc_vector, input_2d_image, input_n_ad_matrix],
        outputs=prediction)

    print(r'model initialization done.')

    return model
