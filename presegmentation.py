import numpy as np
from scipy import spatial
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from findpeaks import findpeaks

def presegmentation(ref_pcd, atl03, num_sampling, radius, quantile, thre_gradient):

    gradient = []

    output = [[] for i in range(ref_pcd.shape[0])]

    tree = spatial.cKDTree(ref_pcd)

    i = 0
    for point in ref_pcd:

        index = tree.query_ball_point(point, radius, return_sorted=True)
        if len(index) >= num_sampling:
            for ele in ref_pcd[index[:num_sampling]]:
                output[i].append(ele)
        else:
            ratio = 1.1
            flag = True
            while flag:
                index_n = tree.query_ball_point(point, radius * ratio, return_sorted=True)
                if len(index_n) >= num_sampling:
                    for ele in ref_pcd[index_n[:num_sampling]]:
                        output[i].append(ele)
                    flag = False
                else:
                    ratio += 0.1

        i += 1

    output = np.array(output)

    for patch in output:
        a1 = patch[:, 0]
        d1 = patch[:, 2]
        a_ = np.array(a1).reshape(-1, 1)
        d_ = np.array(d1).reshape(-1, 1)
        poly_reg = PolynomialFeatures(degree=1)
        X_ploy = poly_reg.fit_transform(a_)
        model = LinearRegression()
        model.fit(X_ploy, d_)
        pre = model.predict(X_ploy)
        gradient.append(model.coef_[0][-1])

    gradient = np.array(gradient)

    index_n = []
    for i in range(len(gradient)):
        if np.abs(gradient[i]) > thre_gradient:
            index_n.append(i)
    index_n = np.array(index_n)

    location = zip(ref_pcd[:, 0][index_n], ref_pcd[:, 2][index_n])
    df = pd.DataFrame(list(location))

    from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

    bandwidth = estimate_bandwidth(df, quantile=quantile)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit_predict(df)

    label_c = ms.labels_

    M = np.max(label_c) + 1

    X_c_, Z_c_ = [[] for l in range(M)], [[] for l in range(M)]

    for data, num in zip(label_c, range(0, len(label_c) + 1)):
        X_c_[data].append(ref_pcd[:, 0][index_n][num])
        Z_c_[data].append(ref_pcd[:, 2][index_n][num])

    X_r = []
    for i in range(M):
        fp = findpeaks(method='peakdetect',interpolate=True)# optional
        results = fp.fit(Z_c_[i])
        peaks_set = np.array(results['df'].iloc[:, -1])
        length = len(np.where(peaks_set == True)[0])
        print(length)
        if length <= 1:
            X_r.append(X_c_[i])
        else:
            location = zip(X_c_[i], Z_c_[i])
            df = pd.DataFrame(list(location))
            km = KMeans(n_clusters=length,
                        init='k-means++',  # random，k-means++
                        n_init=10,
                        max_iter=100,
                        tol=1e-2,
                        random_state=0)
            y_km = km.fit_predict(df)
            label_n = km.labels_
            M_n = np.max(label_n) + 1
            X_c_n = [[] for l in range(M_n)]
            for data, num in zip(label_n, range(0, len(label_n) + 1)):
                X_c_n[data].append(X_c_[i][num])
            for i in range(M_n):
                X_r.append(X_c_n[i])

    scene_boundary = []
    for i in range(len(X_r)):
        poi_min = np.min(X_r[i])
        poi_max = np.max(X_r[i])
        scene_boundary.append([poi_min, poi_max])

    scene_boundary = np.array(scene_boundary)
    arrIndex = np.array(scene_boundary[:, 0]).argsort()
    scene_boundary = scene_boundary[arrIndex]
    scene_boundary = scene_boundary.tolist()
    scenes = []
    for i in range(len(scene_boundary)):
        patches_n = []
        for xyz in atl03:
            if xyz[0] >= scene_boundary[i][0] and xyz[0] <= scene_boundary[i][-1]:
                patches_n.append([xyz[0], xyz[1], xyz[2]])
        patches_n = np.array(patches_n)
        if patches_n.shape[0] != 0:
            scenes.append(patches_n)

    print(r'presegmentation done.')

    return scenes
