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

TF_SEED = 0
tf.random.set_seed(TF_SEED)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def test_model():

    # Load a model
    model = models.load_model(r'saved_model/scn.h5', custom_objects={'GATConv': GATConv,'DMoNPool': DMoNPool})

    fn_lst = os.listdir(r'test_data')

    for name in fn_lst[:]:

        num_neighbors = 10
        std_ratio = 0.1
        radius = 15 # Ball query parameter
        quantile = 0.2 # Clustering parameter
        voxel_size = 30
        num_samples = 15
        threshold = 0.02 # Gradient threshold

        path = r'test_data//' + str(name)

        df = pd.read_csv(path, names=['x', 'y', 'z'], sep=',')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(df.iloc[:, :3]))

        pcd_denoised, _ = pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
        atl03 = np.asarray(pcd_denoised.points)

        pcd_downsampled = pcd_denoised.voxel_down_sample(voxel_size=voxel_size)
        ref_pcd = np.asarray(pcd_downsampled.points)

        scene_lst = presegmentation.presegment_atl03(ref_pcd, atl03, num_samples, radius, quantile, threshold)

        pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, \
        adj_mat_normalized_lst, dc_vec_lst = [], [], [], [], [], []

        for scene in scene_lst:

            # 3D point cloud
            pcd = generate_pcd(scene, 2048)
            pcd_lst.append(pcd)

            # Normal vector
            normal_vec = generate_normal_vec(pcd)
            normal_vec_lst.append(normal_vec)

            # 2D image
            img = generate_img(scene, 128, 512)
            img_lst.append(img)

            # 2D graph
            adj_mat_dense, adj_mat_normalzied, dc_vec = generate_graph(img, 16, 64, 1, 3, 0.315)

            adj_mat_lst.append(adj_mat_dense)
            adj_mat_normalized_lst.append(adj_mat_normalzied)
            dc_vec_lst.append(dc_vec)

        pcd_lst = np.array(pcd_lst)
        normal_vec_lst = np.array(normal_vec_lst)
        img_lst = np.array(img_lst) / 255.
        adj_mat_lst = np.array(adj_mat_lst)
        adj_mat_normalized_lst = np.array(adj_mat_normalized_lst)
        dc_vec_lst = np.array(dc_vec_lst)

        y_pred = model.predict(
            [pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, adj_mat_normalized_lst, dc_vec_lst])

        # Predict scene labels
        label_lst = []
        for pred in y_pred:
            if pred > 0.5:
                label_lst.append(1)
            else:
                label_lst.append(0)
        print(y_pred)

test_model()
