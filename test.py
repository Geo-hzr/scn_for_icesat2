import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

NUM_NEIGHBORS = 10
STD_RATIO = 0.1
RADIUS = 15  # Parameter of ball query
QUANTILE = 0.2  # Parameter of clustering
VOXEL_SIZE = 30
NUM_SAMPLES = 15
THRESHOLD = 0.02  # Threshold of gradient

def test_model():

    # Load a model
    model = models.load_model(r'saved_model/scn.h5', custom_objects={'GATConv': GATConv,'DMoNPool': DMoNPool})

    name_lst = os.listdir(r'test_data')

    for name in name_lst[:]:

        df = pd.read_csv(r'test_data//' + str(name), names=['x', 'y', 'z'], sep=',')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(df.iloc[:, :3]))

        denoised_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=NUM_NEIGHBORS, std_ratio=STD_RATIO)
        atl03 = np.asarray(denoised_pcd.points)

        downsampled_pcd = denoised_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        ref_pcd = np.asarray(downsampled_pcd.points)

        scene_lst = presegmentation.presegment_atl03(ref_pcd, atl03, NUM_SAMPLES, RADIUS, QUANTILE, THRESHOLD)

        pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, normalized_adj_mat_lst, dc_vec_lst = [], [], [], [], [], []

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
            dense_adj_mat, normalzied_adj_mat, dc_vec = generate_graph(img, 16, 64, 1, 3, 0.315)

            adj_mat_lst.append(dense_adj_mat)
            normalized_adj_mat_lst.append(normalzied_adj_mat)
            dc_vec_lst.append(dc_vec)

        pcd_lst = np.array(pcd_lst)
        normal_vec_lst = np.array(normal_vec_lst)
        img_lst = np.array(img_lst) / 255.
        adj_mat_lst = np.array(adj_mat_lst)
        normalized_adj_mat_lst = np.array(normalized_adj_mat_lst)
        dc_vec_lst = np.array(dc_vec_lst)

        y_pred = model.predict([pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, normalized_adj_mat_lst, dc_vec_lst])

        # Predict scene labels
        label_lst = []
        for pred in y_pred:
            if pred > 0.5:  # Threshold of prediction
                label_lst.append(1)
            else:
                label_lst.append(0)
        print(label_lst)

test_model()
