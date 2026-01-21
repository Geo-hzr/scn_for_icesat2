import os
import pandas as pd
import numpy as np
import networkx as nx
from numba import jit
import matplotlib.pyplot as plt
from spektral.utils import normalized_adjacency
import cv2
import open3d as o3d
import PIL.Image as Image
from scipy.spatial.transform import Rotation

def fps(pcd, num_samples):

    pcd = np.expand_dims(pcd, axis=0)
    batch_size, num_points, _ = pcd.shape
    centroid_mat = np.zeros((batch_size, num_samples), dtype=np.long)
    dist_mat = np.ones((batch_size, num_points)) * 1e10
    farthest_idx = np.random.randint(0, num_points, (batch_size,), dtype=np.long)
    batch_idx = np.arange(batch_size)
    for i in range(num_samples):
        centroid_mat[:, i] = farthest_idx
        centroid_pt = pcd[batch_idx, farthest_idx, :].reshape(batch_size, 1, 3)
        dist = np.sum((pcd - centroid_pt) ** 2, -1)
        mask = dist < dist_mat
        dist_mat[mask] = dist[mask]
        farthest_idx = np.argmax(dist_mat[batch_idx])

    return centroid_mat

def normalize_pcd(pcd):

    centroid = np.mean(pcd, axis=0)
    pcd = pcd - centroid
    factor = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
    pcd_normalized = pcd / factor

    return pcd_normalized

def generate_pcd(atl03, num_points):

    centroid_mat = fps(atl03, num_points)
    pcd = normalize_pcd(atl03[centroid_mat][0])

    return pcd

def generate_normal_vec(pcd):

    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(np.array(pcd))
    pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN())

    return np.asarray(pcd_temp.normals)

def generate_img(pcd, img_height, img_width):

    fig = plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.scatter(pcd[:, 0], pcd[:, 2], color='black', s=0.01)
    plt.xlim((np.min(pcd[:, 0]), np.max(pcd[:, 0])))
    plt.ylim((np.min(pcd[:, 2]), np.max(pcd[:, 2])))
    plt.axis('off')
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    buf = np.roll(buf, 3, axis=2)
    img = Image.frombytes('RGB', (w, h), buf.tostring())
    img = np.asarray(img)
    plt.close()

    return img

@jit(nopython=True)
def detect_graph_edge(graph_height, graph_width, px_lst, r_coef, t_coef):

    height_lst = []
    width_lst = []
    for i in range(graph_height):
        for j in range(graph_width):
            height_lst.append(i)
            width_lst.append(j)

    dist_mat = np.zeros((graph_height * graph_width, graph_height * graph_width))
    pair_lst = []
    weight_lst = []

    for i in range(graph_height * graph_width):
        for j in range(i, graph_height * graph_width):
            d = np.sqrt(np.square(height_lst[i] - height_lst[j]) + np.square(width_lst[i] - width_lst[j]))
            dist_mat[i][j] = d
            w = (np.square(d) + (np.square(r_coef) * np.abs(px_lst[i] - px_lst[j]) / 255)) / (2 * np.square(r_coef))
            if d <= r_coef and w <= t_coef:
                pair_lst.append([i, j])
                weight_lst.append(w)

    return pair_lst, dist_mat, weight_lst

def construct_graph(img, r_coef, t_coef):

    px_lst = []
    img_height, img_width = img.shape[0], img.shape[1]

    graph = nx.Graph()
    node_idx = 0
    for i in range(img_height):
        for j in range(img_width):
            graph.add_node(node_idx, pos=(j, img_width - i))
            px_lst.append(np.int(img[i][j]))
            node_idx += 1

    px_lst = np.array(px_lst)

    pair_lst, dist_mat, weight_lst = detect_graph_edge(img_height, img_width, px_lst, r_coef, t_coef)

    for pair, weight in zip(pair_lst, weight_lst):
        graph.add_edge(pair[0], pair[1], weight=weight)

    return graph

def generate_graph(img, graph_height, graph_width, num_channels, r_coef, t_coef):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (graph_width, graph_height))
    graph = construct_graph(np.array(img_gray, dtype=np.uint8), r_coef, t_coef)
    dc = nx.degree_centrality(graph)
    adj_mat = nx.adjacency_matrix(graph)
    adj_mat_dense = adj_mat.todense()
    adj_mat_normalized = normalized_adjacency(adj_mat_dense)
    dc_vec = np.array(list(dc.values())).reshape((graph_height * graph_width, num_channels))

    return adj_mat_dense, adj_mat_normalized, dc_vec

def construct_feature_space(num_points=2048, num_features=3, img_height=128, img_width=512, graph_height=16, graph_width=64, num_channels=1, r_coef=3, t_coef=0.315, src_path_negative=r'train_data/negative', src_path_positive=r'train_data/positive'):

    name_lst_negative = os.listdir(src_path_negative)

    name_lst_positive = os.listdir(src_path_positive)

    pcd_lst, normal_vec_lst, \
    adj_mat_lst, adj_mat_normalized_lst, dc_vec_lst, \
    img_lst, label_lst = [], [], [], [], [], [], []

    class_label = 0

    for name in name_lst_negative[:]:

        label_lst.append(class_label)

        df = pd.read_csv(src_path_negative + r'//' + str(name[:]), names=['x', 'y', 'z'], sep=',')

        atl03 = np.array(df.iloc[:, :num_features])

        # 3D point cloud
        pcd = generate_pcd(atl03, num_points)
        pcd_lst.append(pcd)

        # Normal vector
        normal_vec = generate_normal_vec(pcd)
        normal_vec_lst.append(normal_vec)

        # 2D img
        img = generate_img(atl03, img_height, img_width)
        img_lst.append(img)

        # 2D graph
        adj_mat_dense, adj_mat_normalized, dc_vec = generate_graph(img, graph_height, graph_width, num_channels, r_coef, t_coef)

        adj_mat_lst.append(adj_mat_dense)
        adj_mat_normalized_lst.append(adj_mat_normalized)
        dc_vec_lst.append(dc_vec)

    # Optional
    for name in name_lst_negative[:]:

        label_lst.append(class_label)

        df = pd.read_csv(src_path_negative + r'//' + str(name[:]), names=['x', 'y', 'z'], sep=',')

        atl03 = np.array(df.iloc[:, :num_features])

        r = Rotation.random()
        r_mat = r.as_matrix()
        pcd_rotated = np.dot(r_mat, np.transpose(atl03))
        atl03 = np.transpose(pcd_rotated)

        # 3D point cloud
        pcd = generate_pcd(atl03, num_points)
        pcd_lst.append(pcd)

        # Normal vector
        normal_vec = generate_normal_vec(pcd)
        normal_vec_lst.append(normal_vec)

        # 2D image
        img = generate_img(atl03, img_height, img_width)
        img_lst.append(img)

        # 2D graph
        adj_mat_dense, adj_mat_normalized, dc_vec = generate_graph(img, graph_height, graph_width, num_channels, r_coef, t_coef)

        adj_mat_lst.append(adj_mat_dense)
        adj_mat_normalized_lst.append(adj_mat_normalized)
        dc_vec_lst.append(dc_vec)

    class_label = 1

    for name in name_lst_positive[:]:

        label_lst.append(class_label)

        df = pd.read_csv(src_path_positive + r'//' + str(name[:]), names=['x', 'y', 'z'], sep=',')

        atl03 = np.array(df.iloc[:, :num_features])

        # 3D point cloud
        pcd = generate_pcd(atl03, num_points)
        pcd_lst.append(pcd)

        # Normal vector
        normal_vec = generate_normal_vec(pcd)
        normal_vec_lst.append(normal_vec)

        # 2D image
        img = generate_img(atl03, img_height, img_width)
        img_lst.append(img)

        # 2D graph
        adj_mat_dense, adj_mat_normalized, dc_vec = generate_graph(img, graph_height, graph_width, num_channels, r_coef, t_coef)

        adj_mat_lst.append(adj_mat_dense)
        adj_mat_normalized_lst.append(adj_mat_normalized)
        dc_vec_lst.append(dc_vec)

    # Optional
    for name in name_lst_positive[:]:

        label_lst.append(class_label)

        df = pd.read_csv(src_path_positive + r'//' + str(name[:]), names=['x', 'y', 'z'], sep=',')

        atl03 = np.array(df.iloc[:, :num_features])

        r = Rotation.random()
        r_mat = r.as_matrix()
        pcd_rotated = np.dot(r_mat, np.transpose(atl03))
        atl03 = np.transpose(pcd_rotated)

        # 3D point cloud
        pcd = generate_pcd(atl03, num_points)
        pcd_lst.append(pcd)

        # Normal vector
        normal_vec = generate_normal_vec(pcd)
        normal_vec_lst.append(normal_vec)

        # 2D img
        img = generate_img(atl03, img_height, img_width)
        img_lst.append(img)

        # 2D graph
        adj_mat_dense, adj_mat_normalized, dc_vec = generate_graph(img, graph_width, graph_height, num_channels, r_coef, t_coef)

        adj_mat_lst.append(adj_mat_dense)
        adj_mat_normalized_lst.append(adj_mat_normalized)
        dc_vec_lst.append(dc_vec)

    print(r'Feature augmentation done.')

    return pcd_lst, normal_vec_lst, img_lst, adj_mat_lst, adj_mat_normalized_lst, dc_vec_lst, label_lst
