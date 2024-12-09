from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

import open3d as o3d

# pip
# install
# open3d == 0.12
# .0
#
# !pip
# install
# chart_studio

import chart_studio.plotly as py
import plotly.graph_objs as go

import numpy as np
import open3d as o3d
import copy
import time
import pandas as pd

import os

from sklearn.neighbors import NearestNeighbors
# from open3d.j_visualizer import JVisualizer
# from google.colab import output
# output.enable_custom_widget_manager()
# from google.colab import drive

from scipy.spatial import cKDTree as KDTree

# drive.mount('/content/drive')

ply_a = '/content/drive/MyDrive/fountain_a.ply'
ply_b = '/content/drive/MyDrive/fountain_b.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/bunny_part1.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/bunny_part2.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/01-room_source.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/01-room_target.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/dragon3 - Cloud-other.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/dragon3 - Cloud.ply'

# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/444.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/555.ply'

# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/las-aftercolor-555.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/las-aftercolor-444-withoutcolor-ver2.ply'

# 重要的典型使用场景-树木和建筑物和道路
ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/las-aftercolor-777777-3 - Cloud - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/las-aftercolor-777777-2 - Cloud - Cloud.ply'

# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-16-59-324 - Cloud-222.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-16-33-559 - Cloud-222.ply'

# 重要的典型使用场景-老北区操场附近区域-华测(处理后)同一批次点云成果
ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-17-26-341 - Cloud-clip.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-16-59-324 - Cloud-clip.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-12-32-05-599 - Cloud2 - Cloud - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-12-33-40-531 - Cloud2 - Cloud - Cloud.ply'


ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/dragon3 - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/dragon3 - Cloud-other.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/dragon3 - Cloud - Cloud333.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/dragon3 - Cloud - Cloud222.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/bunny_part1.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/bunny_part2.ply'

# # 北区操场附近的足球场
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-16-59-324 - Cloud-clip222.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-17-26-341 - Cloud-clip222.ply'
# 新北教学楼附近的大楼
ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-12-33-40-531 - Cloud-clip - Cloud - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-12-32-05-599 - Cloud-clip - Cloud - Cloud.ply'
#
#
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-02-27-342 - Cloud - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/202278_03_51_52_000-13-01-45-270 - Cloud - Cloud.ply'


# 南方数据
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_070737_0002 - Cloud - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_071029_0000 - Cloud - Cloud.ply'
#
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0007 - Cloud222.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_080436_0000 - Cloud222.ply'
#
# 这份数据的速度和精度, 明显比sota超过多倍
ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0006 - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0005 - Cloud.ply'


# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_081959_0011 - Cloud222 - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_081959_0012 - Cloud222 - Cloud.ply'
ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_081959_0011 - Cloud - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_081959_0012 - Cloud - Cloud.ply'


ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0007 - Cloud - Cloud222.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_080436_0000 - Cloud - Cloud222.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0005 - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0006 - Cloud.ply'

# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0007 - Cloud222-newRT3.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0007 - Cloud222-newRT2.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/Road_Reference - Cloud3333 - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/Road_Reference - Cloud2222 - Cloud.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0000 - Cloud444.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0000 - Cloud555.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0005 - Cloud - Cloud222.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0006 - Cloud - Cloud.ply'


ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0005 - Cloud - Cloud.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0006 - Cloud - Cloud222.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0005 - Cloud - Cloud222.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_075026_0006 - Cloud - Cloud222.ply'

ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0000 - Cloud444.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0000 - Cloud333.ply'


ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/bunny_part1.ply'
ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/bunny_part2.ply'

# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_083051_0002 - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_083051_0000 - Cloud.ply'


# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_083051_0001 - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_081959_0012 - Cloud.ply'
#
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0000 - Cloud444.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0000 - Cloud333.ply'
#
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_084032_0001 - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/20230811103556584 - Cloud.ply'
#
# ply_a = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/231024_083051_0002 - Cloud.ply'
# ply_b = r'C:\Users\pc\Desktop\test3\lidar_camera_calibration_point_to_plane-master/data/dataset/data/20230811103749000 - Cloud - Cloud.ply'














"""## Utils"""


def best_fit_transform(A, B):
    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)  # covariance matrix
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # R x A + t = B
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=1000, tolerance=0.001):
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        print("mean_error and prev_error:", mean_error, prev_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


# def reg_geo_vge_reduction(A, B, init_pose=None, max_iterations=1000, tolerance=0.001, subsample_ratio=0.1):
#     # get number of dimensions
#     m = A.shape[1]
#
#     # make points homogeneous, copy them to maintain the originals
#     src = np.ones((m + 1, A.shape[0]))
#     dst = np.ones((m + 1, B.shape[0]))
#     src[:m, :] = np.copy(A.T)
#     dst[:m, :] = np.copy(B.T)
#
#     # apply the initial pose estimation
#     if init_pose is not None:
#         src = np.dot(init_pose, src)
#
#     prev_error = 0
#
#     kdtree = KDTree(dst[:m, :])
#
#     for i in range(max_iterations):
#         # randomly sample points from src
#         valid_indices = np.random.choice(src.shape[1], int(src.shape[1] * subsample_ratio), replace=False)
#         src_subset = src[:, valid_indices]
#
#         # find the nearest neighbors between the current source and destination points
#         distances, indices = kdtree.query(src_subset[:m, :].T)
#
#         # convert indices to global indices
#         global_indices = np.arange(B.shape[0])[indices]
#
#         # remove outliers where a point in src has no matching point in dst
#         valid_global_indices = np.where(np.isin(global_indices, valid_indices))[0]
#         src_subset = src_subset[:, valid_global_indices]
#
#         # compute the transformation between the current source and nearest destination points
#         T, _, _ = best_fit_transform(src_subset[:m, :].T, dst[:m, global_indices].T)
#
#         # update the current source
#         src = np.dot(T, src)
#
#         # check error
#         mean_error = np.mean(distances)
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error
#
#     # calculate final transformation
#     T, _, _ = best_fit_transform(A, src[:m, :].T)
#
#     return T, distances, i

def reg_geo_vge_reduction(A, B, init_pose=None, max_iterations=1000, tolerance=0.001):
    # 获取点云的总点数
    total_points = A.shape[0]
    # 计算要保留的点数（40%）
    # num_points_to_keep = int(total_points * 0.4)
    num_points_to_keep = int(total_points * 0.1)
    # 使用 numpy 的 random.choice 进行均匀采样
    indices = np.random.choice(total_points, num_points_to_keep, replace=False)
    # 根据采样索引获取采样后的点云
    A_sampled = A[indices]
    A = A_sampled


    # get number of dimensions
    m = A.shape[1]

    overlap_threshold = 0.5
    overlap_threshold_Import = 0.3
    print("A and B 的大小:", A.shape[0], B.shape[0])

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        print("distances 的大小:", distances.shape[0])
        print("indices 的大小:", indices.shape[0])

        # paired_points = sorted([(distances[i], indices[i], i) for i in range(0, len(distances))])
        # paired_points_array = np.array(paired_points).astype(int)
        # paired_points_array = paired_points_array[:int(paired_points_array.shape[0] / 2), :]
        # # paired_points_array = paired_points_array[:int(paired_points_array.shape[0] / 3), :]

        # 基于空间距离(建筑物类,道路干路类,湖泊河流类,公园园地类等等人地聚集和流动性)的衰减规律, 调整残差的计算方式，以及剔除异常点的策略.
        # 衰减小,残差取小一点,剔除少一点; 衰减大,残差取大一点,剔除多一点.具体的由GWR权重模型来调控.具体而言,1)根据空间距离和分布特征设置排序权重,2)根据空间距离和分布特征设置剔除比例或权重.
        paired_points = sorted([(distances[i], indices[i], i) for i in range(0, len(distances))])
        paired_points_array = np.array(paired_points).astype(int)
        # 删除src中的点在dst中没有匹配点的异常值 (Remove outliers where a point in src has no matching point in dst)
        valid_indices = np.where(np.isin(np.arange(A.shape[0]), paired_points_array[:, 2]))[0]
        src = src[:, valid_indices]
        paired_points_array = paired_points_array[np.isin(paired_points_array[:, 2], valid_indices)]
        # paired_points_array = paired_points_array[:int(paired_points_array.shape[0] / 2), :]
        # paired_points_array = paired_points_array[:int(paired_points_array.shape[0] /5 * 3), :]
        paired_points_array = paired_points_array[:int(paired_points_array.shape[0] /2), :]


        # overlap_points = np.intersect1d(indices, paired_points_array[:, 1])
        # overlap_count = overlap_points.shape[0]
        # overlap_percent = overlap_count / min(A.shape[0], B.shape[0])
        # #小于20%重叠率的初始化操作继续跳过，不错正常迭代的初始化操作. 相当于要想进入快速动态优选，需要先好的准入和好的准出(空间上的重叠率)。
        # print(f"overlap_percent before- {overlap_percent}")
        # if overlap_percent < 0.20:
        #     continue
        # # print(f"ready to continue- {overlap_percent}")
        # print(f"overlap_percent after- {overlap_percent}")

        overlap_points = np.intersect1d(indices, paired_points_array[:, 1])
        overlap_count = overlap_points.shape[0]
        overlap_percent = overlap_count / min(A.shape[0], B.shape[0])

        # if overlap_percent > overlap_threshold_Import:
        #     paired_points_array = paired_points_array[:int(paired_points_array.shape[0] / 2), :]
        #     T, _, _ = best_fit_transform(src[:m, paired_points_array[:, 2]].T, dst[:m, paired_points_array[:, 1]].T)
        #     print("222222", overlap_percent)
        # else:
        #     # T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, :].T)
        #     T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        #     print("333333", overlap_percent)

        # if overlap_percent <= 0.2:
        #     paired_points_array = paired_points_array[:int(paired_points_array.shape[0] / 2), :]
        #     T, _, _ = best_fit_transform(src[:m, paired_points_array[:, 2]].T, dst[:m, paired_points_array[:, 1]].T)
        #     print("222222", overlap_percent)
        # else:
        #     # T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, :].T)
        #     T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        #     print("333333", overlap_percent)

        # if overlap_percent <= 0.2:
        #     T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        # else:
        #     paired_points_array = paired_points_array[:int(paired_points_array.shape[0] / 2), :]
        #     T, _, _ = best_fit_transform(src[:m, paired_points_array[:, 2]].T, dst[:m, paired_points_array[:, 1]].T)

        # # 计算点云的法向量和曲率
        # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # point_cloud.estimate_curvature()
        # # 获取点云的法向量和曲率
        # normals = point_cloud.normals
        # curvatures = point_cloud.curvatures
        # # 可视化点云和曲率
        # o3d.visualization.draw_geometries([point_cloud])
        # # 打印点云中第一个点的法向量和曲率
        # print("法向量大小：", normals[0])
        # print("曲率大小：", curvatures[0])


        T, _, _ = best_fit_transform(src[:m, paired_points_array[:, 2]].T, dst[:m, paired_points_array[:, 1]].T)


        # # compute the transformation between the current source and nearest destination points
        # T, _, _ = best_fit_transform(src[:m, paired_points_array[:, 2]].T, dst[:m, paired_points_array[:, 1]].T)

        # update the current source
        src = np.dot(T, src)

        # # check error
        # mean_error = np.mean(distances)
        # if np.abs(prev_error - mean_error) < tolerance:
        #     break
        # prev_error = mean_error

        # overlap_points = np.intersect1d(indices, paired_points_array[:, 1])
        # overlap_count = overlap_points.shape[0]
        # overlap_percent = overlap_count / min(A.shape[0], B.shape[0])

        length = np.size(paired_points_array)
        print("indices and paired_points_array 的大小:", indices.shape[0], length)
        print("overlap_count 个数:", overlap_count)

        # check error
        mean_error = np.mean(distances)
        print(f"overlap_percent - {overlap_percent}")
        # print(f"mean_error and prev_error - {mean_error}")
        print("mean_error and prev_error:", mean_error, prev_error)
        # 检查重叠百分比和误差
        # if overlap_percent >= overlap_threshold and np.abs(prev_error - mean_error) < tolerance:
        # if np.abs(prev_error - mean_error) < tolerance and i > 800:
        if np.abs(prev_error - mean_error) < tolerance:
            break

        # 更新误差和重叠百分比
        prev_error = mean_error
        prev_overlap = overlap_percent





    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


pcd_a = o3d.io.read_point_cloud(ply_a)
pcd_b = o3d.io.read_point_cloud(ply_b)


def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                                      marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=triangles[:, 0],
                                j=triangles[:, 1], k=triangles[:, 2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()


def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)

    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


from sklearn.metrics import mean_squared_error


def get_metrics(T, R, t):
    R_measure = T[:3, :3]
    T_measure = T[:3, 3]
    eul_measure = rot2eul(R_measure)
    eul_def = rot2eul(R)
    ang_dist = np.linalg.norm(eul_def - eul_measure)
    mse_translation = mean_squared_error(T_measure, t)
    return ang_dist, mse_translation


def rot2eul(R):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


dim = 3  # number of dimensions of the points
noise_sigma = .01  # standard deviation error to be added
translation = .4  # max translation of the test set
rotation = .4
N = 10  # number of random points in the dataset


def pcd_changes(pcd, translation):
    new_pcd = copy.deepcopy(pcd)

    # Translate
    new_pcd = new_pcd.translate(translation)

    # Rotate
    R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
    new_pcd.rotate(R, center=(0, 0, 0))

    # Add noise
    new_pcd_points = np.asarray(new_pcd.points)
    new_pcd_points += np.random.randn(new_pcd_points.shape[0], new_pcd_points.shape[1]) * noise_sigma
    new_pcd.points = o3d.utility.Vector3dVector(new_pcd_points)

    return new_pcd, R


def my_print(header, x):
    print(f"{header}:\n")
    try:
        for row in x:
            print(' '.join(map(lambda x: "{:.3f}\t".format(x), row)))
    except:
        print(np.array2string(x, formatter={'float_kind': lambda x: "%.2f" % x}))



# 定义一个函数，用于加载las文件并转换为open3d的点云格式
def load_las_file(file_name):
    # pcd = o3d.io.read_point_cloud(file_name, format='las')
    pcd = o3d.io.read_point_cloud(file_name, format='ply')
    return pcd

# 定义一个函数，用于执行ICP匹配
def execute_icp(source_pcd, target_pcd, threshold=0.02, trans_init=[0, 0, 0, 0, 0, 0]):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p



"""## Visualization

### Original
"""

# o3d.visualization.draw_geometries = draw_geometries  # replace function
o3d.visualization.draw_geometries([pcd_a + pcd_b])

"""### After transformation"""

# pcd_b_changed, R = pcd_changes(pcd_b, [0.05, 0.05, 0.05])
pcd_b_changed = pcd_b

# o3d.visualization.draw_geometries = draw_geometries  # replace function
# o3d.visualization.draw_geometries([pcd_a + pcd_b_changed])

"""### geo_vge_reductionP"""

start = time.time()
T, distances, iterations = reg_geo_vge_reduction(np.asarray(pcd_b_changed.points), np.asarray(pcd_a.points), tolerance=0.000001)
print(time.time() - start)
print(f"distances - {distances}")
print(f"iterations - {iterations}")

pcd_b_geo_vge_reduction = copy.deepcopy(pcd_b_changed).transform(T)

# o3d.visualization.draw_geometries = draw_geometries  # replace function
# o3d.visualization.draw_geometries([pcd_a + pcd_b_geo_vge_reduction])
# o3d.visualization.draw_geometries([pcd_a + pcd_b_geo_vge_reduction], window_name="[pcd_a + pcd_b_geo_vge_reduction]", width=880, height=1290)
o3d.visualization.draw_geometries([pcd_a + pcd_b_geo_vge_reduction], window_name="pcd_a + pcd_b_geo_vge_reduction", width=1900, height=1000)






# 设置工作目录到包含las文件的目录
# 请根据实际情况修改下面的路径
# work_dir = "/path/to/your/las/files"
work_dir = r"C:/Users/pc/Desktop/test3/lidar_camera_calibration_point_to_plane-master/data/dataset/data/test_save_las_files"

os.chdir(work_dir)

# 获取所有las文件
# las_files = [f for f in os.listdir(work_dir) if f.endswith('.las')]
las_files = [f for f in os.listdir(work_dir) if f.endswith('.ply')]

# 对文件进行排序，确保它们是按顺序处理的
las_files.sort()


if len(las_files) == 2:
    print("数组中的元素数量等于2")

    # 遍历文件列表，两两配对执行ICP匹配
    for i in range(len(las_files) - 1):
        source_pcd = load_las_file(las_files[i])
        target_pcd = load_las_file(las_files[i + 1])

        # 拷贝target_pcd点云数据, 后面的变换矩阵T直接用这个拷贝数据
        copied_target_pcd = o3d.geometry.PointCloud()
        copied_target_pcd.points = target_pcd.points
        copied_target_pcd.colors = target_pcd.colors  # 如果有颜色数据，也进行复制
        # copied_pcd.normals = target_pcd.normals  # 如果有法线数据，也进行复制


        # # 执行ICP匹配
        # reg_p2p = execute_icp(source_pcd, target_pcd)

        """### Algorithms"""
        """### ICP"""
        # start = time.time()
        # T, distances, iterations = icp(np.asarray(target_pcd.points), np.asarray(source_pcd.points), tolerance=0.000001)
        # print(time.time() - start)
        # print(f"distances - {distances}")
        # print(f"iterations - {iterations}")
        # pcd_b_icp = copy.deepcopy(pcd_b_changed).transform(T)
        # o3d.visualization.draw_geometries([pcd_a + pcd_b_icp])
        """### Tr-ICP"""
        start = time.time()
        T, distances, iterations = reg_geo_vge_reduction(np.asarray(target_pcd.points), np.asarray(source_pcd.points), tolerance=0.000001)
        # T, distances, iterations = reg_geo_vge_reduction(np.asarray(target_pcd.points), np.asarray(source_pcd.points), None, 100, tolerance=0.000001)
        print(time.time() - start)
        print(f"distances - {distances}")
        print(f"iterations - {iterations}")
        # reg_p2p = copy.deepcopy(target_pcd).transform(T)
        reg_p2p = copy.deepcopy(copied_target_pcd).transform(T)
        # o3d.visualization.draw_geometries([pcd_a + pcd_b_tricp])
        o3d.visualization.draw_geometries([source_pcd + reg_p2p])

        # # 应用变换
        # source_pcd = source_pcd.transform(reg_p2p.transformation)

        # 计算平均值
        average_distance = np.mean(distances)
        # 判断平均值是否小于10.0cm,超过阈值本次不会保存
        # if average_distance < 10.0:
        if average_distance < 12.0:
            print(f"Average distance {average_distance} is less than 12.0, save and write result.")
            # 保存匹配后的点云数据
            save_path = f"{las_files[i]}_matched_to_{las_files[i + 1]}.ply"
            # o3d.io.write_point_cloud(save_path, source_pcd, format='ply')
            write_result = o3d.io.write_point_cloud(save_path, reg_p2p, format='ply')
            print(f"write succeed222 - {write_result}")
        else:
            print(f"Average distance {average_distance} is not less than 12.0, program continues.")




if len(las_files) > 2:
    print("数组中的元素数量大于2")

    # 遍历文件列表，两两配对以及与邻居的邻居配对执行ICP匹配
    for i in range(len(las_files) - 2):  # 循环到倒数第三个文件
        source_pcd = load_las_file(las_files[i])
        target_pcd1 = load_las_file(las_files[i + 1])
        target_pcd2 = load_las_file(las_files[i + 2])


        """### Algorithms"""
        """### Tr-ICP"""
        start = time.time()
        T, distances, iterations = reg_geo_vge_reduction(np.asarray(target_pcd1.points), np.asarray(source_pcd.points), tolerance=0.000001)
        # T, distances, iterations = reg_geo_vge_reduction(np.asarray(target_pcd1.points), np.asarray(source_pcd.points), None, 100, tolerance=0.000001)

        print(time.time() - start)
        print(f"distances - {distances}")
        print(f"iterations - {iterations}")
        reg_p2p_1 = copy.deepcopy(target_pcd1).transform(T)
        # o3d.visualization.draw_geometries([source_pcd + reg_p2p_1])
        # 计算平均值
        average_distance_1 = np.mean(distances)

        """### Tr-ICP"""
        start = time.time()
        T, distances, iterations = reg_geo_vge_reduction(np.asarray(target_pcd2.points), np.asarray(source_pcd.points), tolerance=0.000001)
        # T, distances, iterations = reg_geo_vge_reduction(np.asarray(target_pcd2.points), np.asarray(source_pcd.points), None, 100, tolerance=0.000001)
        print(time.time() - start)
        print(f"distances - {distances}")
        print(f"iterations - {iterations}")
        reg_p2p_2 = copy.deepcopy(target_pcd2).transform(T)
        # o3d.visualization.draw_geometries([source_pcd + reg_p2p_2])
        # 计算平均值
        average_distance_2 = np.mean(distances)

        # 只保留结果最好的一份点云数据
        if average_distance_2 < average_distance_1:
            if average_distance_2 < 8.0:
                print(f" average_distance_2 <= average_distance_1 ")
                print(f"Average distance {average_distance_2} is less than 8.0, save and write result.")
                # 保存匹配后的点云数据
                save_path = f"{las_files[i]}_matched_to_{las_files[i + 2]}.ply"
                write_result = o3d.io.write_point_cloud(save_path, reg_p2p_2, format='ply')
                print(f"write succeed222 - {write_result}")
            else:
                print(f"Average distance {average_distance_2} is not less than 8.0, program continues.")
        else:
            if average_distance_1 < 8.0:
                print(f" average_distance_2 > average_distance_1 ")
                print(f"Average distance {average_distance_1} is less than 8.0, save and write result.")
                # 保存匹配后的点云数据
                save_path = f"{las_files[i]}_matched_to_{las_files[i + 1]}.ply"
                write_result = o3d.io.write_point_cloud(save_path, reg_p2p_1, format='ply')
                print(f"write succeed222 - {write_result}")
            else:
                print(f"Average distance {average_distance_1} is not less than 8.0, program continues.")



















