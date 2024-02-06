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

def farthest_point_sample(pts, num):

    pc1 = np.expand_dims(pts, axis=0)  # 1, N, 3
    batchsize, npts, dim = pc1.shape
    centroids = np.zeros((batchsize, num), dtype=np.long)
    distance = np.ones((batchsize, npts)) * 1e10
    farthest_id = np.random.randint(0, npts, (batchsize,), dtype=np.long)
    batch_index = np.arange(batchsize)
    for i in range(num):
        centroids[:, i] = farthest_id
        centro_pt = pc1[batch_index, farthest_id, :].reshape(batchsize, 1, 3)
        dist = np.sum((pc1 - centro_pt) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest_id = np.argmax(distance[batch_index])

    return centroids

def norm_point_cloud(pcd):

    centroid = np.mean(pcd, axis=0)
    pc = pcd - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m

    return pc_normalized

@jit(nopython=True)
def edge_detection(m, n, I, r, t):

    L = 255
    X = []
    Y = []
    for i in range(m):
        for j in range(n):
            X.append(i)
            Y.append(j)

    dis = np.zeros((m * n, m * n))
    v = []
    weight_v = []

    for i in range(m * n):
        for j in range(i, m * n):
            d = np.sqrt(np.square(X[i] - X[j]) + np.square(Y[i] - Y[j]))
            dis[i][j] = d
            w = (np.square(d) + (np.square(r) * np.abs(I[i] - I[j]) / L)) / (2 * np.square(r))
            if d <= r and w <= t:
                v.append([i, j])
                weight_v.append(w)

    return v, dis, weight_v

def graph_construction(img, r, t):

    I = []
    m, n = img.shape[0], img.shape[1]

    G = nx.Graph()
    node_num = 0
    for i in range(m):
        for j in range(n):
            G.add_node(node_num, pos=(j, n - i))
            I.append(np.int(img[i][j]))
            node_num += 1

    I = np.array(I)

    g, dis_m, w = edge_detection(m, n, I, r, t)

    for data, weights in zip(g, w):
        G.add_edge(data[0], data[1], weight=weights)

    return G

def point_cloud_conversion(data, num_points):

    centroids = farthest_point_sample(data, num_points)
    qbp_raw_data = data[centroids][0]
    downpcd = norm_point_cloud(qbp_raw_data)

    return downpcd

def normal_vector_conversion(downpcd):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(downpcd))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN())

    return np.asarray(pcd.normals)

def image_conversion(data,img_width,img_height):

    fig = plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.scatter(data[:, 0], data[:, 2], color='black', s=0.01)
    plt.xlim((min(data[:, 0]), max(data[:, 0])))
    plt.ylim((min(data[:, 2]), max(data[:, 2])))
    plt.axis('off')
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGB", (w, h), buf.tostring())
    image = np.asarray(image)
    plt.close()

    return image

def graph_conversion(image,graph_width,graph_height,num_channels, r, t):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (graph_width, graph_height))
    graph = graph_construction(np.array(gray, dtype=np.uint8), r, t)
    degree = nx.degree_centrality(graph)
    As = nx.adjacency_matrix(graph)
    A = As.todense()
    N_A = normalized_adjacency(A)
    X = np.array(list(degree.values())).reshape((graph_width * graph_height, num_channels))

    return A, N_A, X

def feature_space_construction(num_points=2048,
                    num_features=3,
                    img_width=512,
                    img_height=128,
                    graph_width=64,
                    graph_height=16,
                    num_channels=1,
                    r=3,
                    t=0.315,
                    fp_positive_path=r'train_data/positive',
                    fp_negative_path=r'train_data/negative',
                    ):

    fp_positive = os.listdir(fp_positive_path)

    fp_negative = os.listdir(fp_negative_path)

    input_point_cloud, input_normal_vector, \
    input_ad_matrix, input_n_ad_matrix, input_dc_vector, \
    input_image, input_label = [], [], [], [], [], [], []

    class_label = 0

    for name in fp_negative[:]:

        input_label.append(class_label)

        df = pd.read_csv(fp_negative_path + r'//' + str(name[:]), names=['x', 'y', 'z'], sep=',')

        data = np.array(df.iloc[:, :num_features])

        # 3d point cloud
        downpcd = point_cloud_conversion(data, num_points)
        input_point_cloud.append(downpcd)

        # normal vector
        normal_vector = normal_vector_conversion(downpcd)
        input_normal_vector.append(normal_vector)

        # 2d image
        image = image_conversion(data,img_width,img_height)
        input_image.append(image)

        # 2d graph
        A, N_A, X = graph_conversion(image,graph_width,graph_height,num_channels,r,t)

        input_ad_matrix.append(A)
        input_dc_vector.append(X)
        input_n_ad_matrix.append(N_A)

    class_label = 1

    for name in fp_positive[:]:

        input_label.append(class_label)

        df = pd.read_csv(fp_positive_path + r'//' + str(name[:])
                         , names=['x', 'y', 'z'], sep=',')

        data = np.array(df.iloc[:, :num_features])

        # 3d point cloud
        downpcd = point_cloud_conversion(data, num_points)
        input_point_cloud.append(downpcd)

        # normal vector
        normal_vector = normal_vector_conversion(downpcd)
        input_normal_vector.append(normal_vector)

        # 2d image
        image = image_conversion(data, img_width, img_height)
        input_image.append(image)

        # 2d graph
        A, N_A, X = graph_conversion(image, graph_width, graph_height, num_channels, r, t)

        input_ad_matrix.append(A)
        input_dc_vector.append(X)
        input_n_ad_matrix.append(N_A)

    print(r'feature augmentation done.')

    return input_point_cloud, input_normal_vector, input_ad_matrix, input_n_ad_matrix, input_dc_vector, input_image, input_label
