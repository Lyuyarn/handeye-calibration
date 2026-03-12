"""

眼在手上 计算得是 相机相对于机械臂末端 齐次变换矩阵
计算 这个矩阵需要得是  标定板相对于相机得次变换矩阵 * 相机相对于机械臂末端得齐次变换矩阵 * 机械臂末端相对于基座得齐次变换矩阵

机械臂末端相对于基座得齐次变换矩阵（也就是机械臂位姿变换得齐次变换矩阵）

"""
import csv

import numpy as np
import os
import logging

from libs.log_setting import CommonLog

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz@Ry@Rx

    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z= pose[:3]
    rx, ry, rz = np.array(pose[3:])*np.pi/180  # 角度转弧度
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H


def save_matrices_to_csv(matrices, file_name):

    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)


def poses_main(tag,error_num):

    file_num = [f for f in os.listdir(tag) if f.endswith('.armpose')]
    lines_list = []
    for i in range(0, len(file_num)):   #标定好的图片在images_path路径下，从0.jpg到x.jpg
        if i not in error_num:
            pose_file = os.path.join(tag,f"{i}.armpose")

            if os.path.exists(pose_file):

                logger_.info(f'读 {pose_file}')

            with open(f"{pose_file}", "r",encoding="utf-8") as f:
                # 读取文件中的所有行
                lines = f.readlines()
            # 定义一个空列表，用于存储结果

            # 遍历每一行数据
            lines = [float(i)  for line in lines for i in line.split(',')]

            lines_list.append(lines)

    matrices = []
    for i in range(0,len(lines_list)):
        matrices.append(pose_to_homogeneous_matrix(lines_list[i]))


    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, f'RobotToolPoseInHand.csv')

