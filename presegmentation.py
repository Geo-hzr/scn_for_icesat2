import numpy as np
from scipy import spatial
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from findpeaks import findpeaks

def presegment_atl03(ref_pcd, atl03, num_samples, radius, quantile, threshold):

    grad_lst = []

    seg_lst = [[] for _ in range(ref_pcd.shape[0])]

    tree = spatial.cKDTree(ref_pcd)

    i = 0
    for src_pt in ref_pcd:

        idx_lst = tree.query_ball_point(src_pt, radius, return_sorted=True)
        if len(idx_lst) >= num_samples:
            for tgt_pt in ref_pcd[idx_lst[:num_samples]]:
                seg_lst[i].append(tgt_pt)
        else:
            ratio = 1.1
            is_extended = True
            while is_extended:
                extended_idx_lst = tree.query_ball_point(src_pt, radius * ratio, return_sorted=True)
                if len(extended_idx_lst) >= num_samples:
                    for tgt_pt in ref_pcd[extended_idx_lst[:num_samples]]:
                        seg_lst[i].append(tgt_pt)
                    is_extended = False
                else:
                    ratio += 0.1

        i += 1

    seg_lst = np.array(seg_lst)

    for seg in seg_lst:
        x = np.array(seg[:, 0]).reshape(-1, 1)
        z = np.array(seg[:, 2]).reshape(-1, 1)
        pf = PolynomialFeatures(degree=1)
        x_transformed = pf.fit_transform(x)
        lr = LinearRegression()
        lr.fit(x_transformed, z)
        _ = lr.predict(x_transformed)
        grad_lst.append(lr.coef_[0][-1])

    grad_lst = np.array(grad_lst)

    idx_lst = []
    for i in range(len(grad_lst)):
        if np.abs(grad_lst[i]) > threshold:
            idx_lst.append(i)
    idx_lst = np.array(idx_lst)

    loc = zip(ref_pcd[:, 0][idx_lst], ref_pcd[:, 2][idx_lst])
    df = pd.DataFrame(list(loc))

    from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

    band_width = estimate_bandwidth(df, quantile=quantile)
    ms = MeanShift(bandwidth=band_width)
    ms.fit_predict(df)

    label_lst = ms.labels_

    num_labels = np.max(label_lst) + 1

    x_clustered, z_clustered = [[] for _ in range(num_labels)], [[] for _ in range(num_labels)]

    for label, num in zip(label_lst, range(0, len(label_lst) + 1)):
        x_clustered[label].append(ref_pcd[:, 0][idx_lst][num])
        z_clustered[label].append(ref_pcd[:, 2][idx_lst][num])

    x_referenced = []
    for i in range(num_labels):
        fp = findpeaks(method='peakdetect', interpolate=True)
        pt_lst = np.array(fp.fit(z_clustered[i])['df'].iloc[:, -1])
        num_points = len(np.where(pt_lst == True)[0])
        if num_points <= 1:
            x_referenced.append(x_clustered[i])
        else:
            loc = zip(x_clustered[i], z_clustered[i])
            df = pd.DataFrame(list(loc))
            km = KMeans(n_clusters=num_points, init='k-means++', n_init=10, max_iter=100, tol=1e-2, random_state=0)
            _ = km.fit_predict(df)
            temp_label_lst = km.labels_
            temp_num_labels = np.max(temp_label_lst) + 1
            x_temp = [[] for _ in range(temp_num_labels)]
            for label, num in zip(temp_label_lst, range(0, len(temp_label_lst) + 1)):
                x_temp[label].append(x_clustered[i][num])
            for j in range(temp_num_labels):
                x_referenced.append(x_temp[j])

    boundary_lst = []
    for i in range(len(x_referenced)):
        x_min = np.min(x_referenced[i])
        x_max = np.max(x_referenced[i])
        boundary_lst.append([x_min, x_max])

    boundary_lst = np.array(boundary_lst)
    arr_idx = np.array(boundary_lst[:, 0]).argsort()
    boundary_lst = boundary_lst[arr_idx]
    boundary_lst = boundary_lst.tolist()

    scene_lst = []
    for i in range(len(boundary_lst)):
        coord_lst = []
        for coord in atl03:
            if coord[0] >= boundary_lst[i][0] and coord[0] <= boundary_lst[i][-1]:
                coord_lst.append([coord[0], coord[1], coord[2]])
        coord_lst = np.array(coord_lst)
        if coord_lst.shape[0] != 0:
            scene_lst.append(coord_lst)

    print(r'Presegmentation done.')

    return scene_lst
