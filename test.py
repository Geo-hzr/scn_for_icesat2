import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from spektral.layers.convolutional import *
from spektral.layers.pooling import *
from tensorflow.keras import *
import presegmentation
from feature_augmentation import *

#fix random seed
tf_seed = 0
tf.random.set_seed(tf_seed)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def test_model():

    #load a model
    model = models.load_model(r'saved_model/scn.h5', custom_objects={'GATConv': GATConv,'DMoNPool': DMoNPool})

    fp = os.listdir(r'test_data')

    for name in fp[:]:

        num_neighbor = 10
        std_ratio = 0.1
        radius = 15 # for ball query
        quantile = 0.2 # for clustering
        voxel_size = 30
        num_sampling = 15
        thre_gradient = 0.02 # adjusted

        # print(name)

        path = r'test_data//' + str(name)

        df = pd.read_csv(path, names=['x', 'y', 'z'], sep=',')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.array(df.iloc[:, :3]))

        pcd, denoised = pcd.remove_statistical_outlier(nb_neighbors=num_neighbor, std_ratio=std_ratio)
        atl03 = np.asarray(pcd.points)

        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downpcd_point = np.asarray(downpcd.points)
        ref_pcd = downpcd_point

        scenes = presegmentation.presegmentation(ref_pcd, atl03, num_sampling, radius, quantile, thre_gradient)

        input_point_cloud, input_normal_vector, \
        input_ad_matrix, input_n_ad_matrix, input_dc_vector, \
        input_image, input_label = [], [], [], [], [], [], []

        for data in scenes:

            # 3d point cloud
            downpcd = point_cloud_conversion(data, 2048)
            input_point_cloud.append(downpcd)

            # normal vector
            normal_vector = normal_vector_conversion(downpcd)
            input_normal_vector.append(normal_vector)

            # 2d image
            image = image_conversion(data, 512, 128)
            input_image.append(image)

            # 2d graph
            A, N_A, X = graph_conversion(image, 64, 16, 1, 3, 0.315)

            input_ad_matrix.append(A)
            input_dc_vector.append(X)
            input_n_ad_matrix.append(N_A)

        input_point_cloud = np.array(input_point_cloud)
        input_normal_vector = np.array(input_normal_vector)
        input_ad_matrix = np.array(input_ad_matrix)
        input_n_ad_matrix = np.array(input_n_ad_matrix)
        input_dc_vector = np.array(input_dc_vector)
        input_image = np.array(input_image) / 255.

        y_pred = model.predict(
            [input_point_cloud, input_normal_vector, input_ad_matrix, input_dc_vector, input_image, input_n_ad_matrix])

        # predicted scene labels
        y_pred_classes = []
        for label in y_pred:
            if label > 0.5:
                y_pred_classes.append(1)
            else:
                y_pred_classes.append(0)
        print(y_pred)

test_model()
