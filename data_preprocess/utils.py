import numpy as np
def speed_acc_sig(move_sig, fp):
    motion_x = []
    motion_y = []
    for motion in move_sig:
        motion_x.append(motion[:, 1].reshape(1, -1))  # 先按照列表的方式，分别记录每一个ROI点的运动信号
        motion_y.append(motion[:, 2].reshape(1, -1))
    motion_x = np.squeeze(np.array(motion_x, dtype=int))  # 将列表转换为数组，并删除维数长度为1的维度
    motion_y = np.squeeze(np.array(motion_y, dtype=int))  # 每一行数据为一个特征点的运动信息

    speed_x = np.zeros(motion_x.shape)
    speed_y = np.zeros(motion_y.shape)
    speed_x[:, 1:] = (motion_x[:, 1:] - motion_x[:, 0:-1]) * fp
    speed_y[:, 1:] = (motion_y[:, 1:] - motion_y[:, 0:-1]) * fp
    # speed_x = speed_x.T
    # speed_y = speed_y.T  # 先转置按照每一列进行最大最小归一化，再转置回来
    # speed_x = ((speed_x - speed_x.min(axis=0)) / (speed_x.max(axis=0) - speed_x.min(axis=0))).T
    # speed_y = ((speed_y - speed_y.min(axis=0)) / (speed_y.max(axis=0) - speed_y.min(axis=0))).T

    acc_x = np.zeros(speed_x.shape)
    acc_y = np.zeros(speed_y.shape)
    acc_x[:, 1:] = (speed_x[:, 1:] - speed_x[:, 0:-1]) * fp
    acc_y[:, 1:] = (speed_y[:, 1:] - speed_y[:, 0:-1]) * fp
    # acc_x = acc_x.T
    # acc_y = acc_y.T
    # acc_x = ((acc_x - acc_x.min(axis=0)) / (acc_x.max(axis=0) - acc_x.min(axis=0))).T
    # acc_y = ((acc_y - acc_y.min(axis=0)) / (acc_y.max(axis=0) - acc_y.min(axis=0))).T
    return speed_x, speed_y, acc_x, acc_y


# 对patches运动加速度求和
def sum_speed_acc_sig(sx, sy, ax, ay, motion):
    sum_sx = np.sum(sx, axis=0).reshape((1, -1))
    sum_sy = np.sum(sy, axis=0).reshape((1, -1))
    sum_ax = np.sum(ax, axis=0).reshape((1, -1))
    sum_ay = np.sum(ay, axis=0).reshape((1, -1))
    sum_motion = np.sum(np.array(motion), axis=0)[:, 1:].T
    return sum_sx, sum_sy, sum_ax, sum_ay, sum_motion