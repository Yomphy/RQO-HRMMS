import padasip as pad
import numpy as np
import scipy.signal._savitzky_golay
from scipy.signal import *
import math

# 仿射投影 AP
def AP(d_sig, x_sig, length, **kwargs):
    return pad.filters.FilterAP(n=length, order=int(kwargs['order']), mu=float(kwargs['miu']), ifc=float(kwargs['fc']), w=str(kwargs['weight'])).run(d=d_sig, x=x_sig

# 用于传入信号、参考信号、滤波器参数、以及滤波器方法直接调用自适应滤波算法的函数
def adap_filter(rppgs, winsize, speed_x, speed_y, speed_z, fps, Method=None, params_x={}, params_y={}, params_z={}):
    # rppgs： 窗信号列表

    adaptive_x_rppgs = []  # 存储x轴运动滤波后期望信号
    adaptive_xy_rppgs = []  # 存储xy轴运动滤波后期望信号
    adaptive_xyz_rppgs = []  # 存储xyz轴运动滤波后期望信号
    notch_rppgs = []
    win_index = 0
    w_size = int(winsize)

    for win_rppgs in rppgs:  # 对每个窗信号做运动消除，patch_bvps是窗信号列表
        patch_index = 0
        win_x_signal = []
        win_xy_signal = []
        win_xyz_signal = []
        win_notch_signal = []
        for patch_rppg in win_rppgs:  # 对于窗信号中的每一个ROI，使用对应的的运动信息
            time_x = int(params_x['time_history'])
            time_y = int(params_y['time_history'])
            time_z = int(params_z['time_history'])
            # 针对每一个窗信号在对应的ROI序号信号中提取相对应的窗运动信息
            # Patches
            # noise_x = speed_x[patch_index, int(win_index * 1 * int(fps)):int((win_index * 1 * fps + w_size * int(fps)))]
            # noise_y = speed_y[patch_index, int(win_index * 1 * int(fps)):int((win_index * 1 * fps + w_size * int(fps)))]

            # Holistic
            noise_x = speed_x[0, int(win_index * 1 * int(fps)):int((win_index * 1 * fps + w_size * int(fps)))]
            noise_y = speed_y[0, int(win_index * 1 * int(fps)):int((win_index * 1 * fps + w_size * int(fps)))]
            noise_z = speed_z[int(win_index * 1 * int(fps)):int((win_index * 1 * fps + w_size * int(fps)))]
            noise_x1 = noise_x.copy(); noise_y1 = noise_y.copy(); noise_z1 = noise_z.copy()
            std_x = np.std(noise_x1);  std_y = np.std(noise_y1);  std_z = np.std(noise_z1)

            noise_x = (noise_x - np.min(noise_x)) / (np.max(noise_x) - np.min(noise_x))  # 归一化
            noise_y = (noise_y - np.min(noise_y)) / (np.max(noise_y) - np.min(noise_y))  # 归一化
            noise_z = (noise_z - np.min(noise_z)) / (np.max(noise_z) - np.min(noise_z))  # 归一化

            noise_x = pad.preprocess.input_from_history(noise_x, time_x)  # 噪声信号转换成合适的输入信号形式
            noise_y = pad.preprocess.input_from_history(noise_y, time_y)  # 噪声信号转换成合适的输入信号形式
            noise_z = pad.preprocess.input_from_history(noise_z, time_z)  # 噪声信号转换成合适的输入信号形式
            patch_rppg = pad.preprocess.standardize(patch_rppg)

            _, e_x, _ = Method(patch_rppg[time_x - 1:], noise_x, length=time_x, **params_x)
            _, e_xy, _ = Method(e_x, noise_y, length=time_y, **params_y)
            _, e_xyz, _ = Method(e_xy, noise_z, length=time_z, **params_z)
            # if std_z >= 8000:
            #     _, e_xyz, _ = Method(e_xy, noise_z, length=time_z, **params_z)
            # else:
            #     e_xyz = e_xy
            #
            # FX, PX = welch(noise_x1, nperseg=256, fs=fps, nfft=2048)  # 采样频率阵列， 功率谱密度
            # FY, PY = welch(noise_y1, nperseg=256, fs=fps, nfft=2048)  # 采样频率阵列， 功率谱密度
            # FZ, PZ = welch(noise_z1, nperseg=256, fs=fps, nfft=2048)  # 采样频率阵列， 功率谱密度
            # FX = FX[0:230].astype(np.float32);  PX = PX[0:230].astype(np.float32)
            # FY = FY[0:230].astype(np.float32);  PY = PY[0:230].astype(np.float32)
            # FZ = FZ[0:230].astype(np.float32);  PZ = PZ[0:230].astype(np.float32)
            #
            # peaks_x, _ = find_peaks(PX, height=max(PX) / 50);  in_x = np.argsort(PX[peaks_x])[-1:]; target_x = peaks_x[in_x]
            # peaks_y, _ = find_peaks(PY, height=max(PY) / 50);  in_y = np.argsort(PY[peaks_y])[-1:]; target_y = peaks_y[in_y]
            # peaks_z, _ = find_peaks(PZ, height=max(PZ) / 50);  in_z = np.argsort(PZ[peaks_z])[-3:]; target_z = peaks_z[in_z]
            #
            # speed_xf = FX[target_x]
            # speed_yf = FY[target_y]
            # speed_zf = FZ[target_z]
            #
            # # 陷波滤波器系统函数系数
            # r = 0.95
            # fs = 30
            #
            # notch_x_rppg = e_xyz
            # for fx in speed_xf:
            #     wn = 2 * math.pi * fx / fs
            #     b = np.array([1, -2 * math.cos(wn), 1])
            #     a = np.array([1, -2 * r * math.cos(wn), r ** 2])
            #     notch_x_rppg = filtfilt(b, a, notch_x_rppg)
            #
            # notch_xy_rppg = notch_x_rppg
            # for fy in speed_yf:
            #     wn = 2 * math.pi * fy / fs
            #     b = np.array([1, -2 * math.cos(wn), 1])
            #     a = np.array([1, -2 * r * math.cos(wn), r ** 2])
            #     notch_xy_rppg = filtfilt(b, a, notch_xy_rppg)
            #
            # notch_xyz_rppg = notch_xy_rppg
            # for fz in speed_zf:
            #     wn = 2 * math.pi * fz / fs
            #     b = np.array([1, -2 * math.cos(wn), 1])
            #     a = np.array([1, -2 * r * math.cos(wn), r ** 2])
            #     notch_xyz_rppg = filtfilt(b, a, notch_xyz_rppg)

            win_x_signal.append(e_x)
            win_xy_signal.append(e_xy)
            win_xyz_signal.append(e_xyz)

            # win_notch_signal.append(notch_xyz_rppg)

            patch_index += 1
        win_index += 1
        adaptive_x_rppgs.append(np.array(win_x_signal))
        adaptive_xy_rppgs.append(np.array(win_xy_signal))
        adaptive_xyz_rppgs.append(np.array(win_xyz_signal))
        # notch_rppgs.append(np.array(win_notch_signal))
    return adaptive_xyz_rppgs


# 每个窗信号会进行动作分类，窗信号进行不同的滤波选择
def adap_filter_motion_class(win_rppg, noise_x, noise_y, noise_z, fps, Method=None, params_x={}, params_y={}, params_z={}):
    time_x = int(params_x['time_history'])
    time_y = int(params_y['time_history'])
    time_z = int(params_z['time_history'])
    # 针对每一个窗信号在对应的ROI序号信号中提取相对应的窗运动信息

    noise_x = (noise_x - np.min(noise_x)) / (np.max(noise_x) - np.min(noise_x))  # 噪声归一化
    noise_y = (noise_y - np.min(noise_y)) / (np.max(noise_y) - np.min(noise_y))  # 噪声归一化
    noise_z = (noise_z - np.min(noise_z)) / (np.max(noise_z) - np.min(noise_z))  # 噪声归一化

    noise_x = pad.preprocess.input_from_history(noise_x, time_x)  # 噪声信号转换成合适的输入信号形式
    noise_y = pad.preprocess.input_from_history(noise_y, time_y)  # 噪声信号转换成合适的输入信号形式
    noise_z = pad.preprocess.input_from_history(noise_z, time_z)  # 噪声信号转换成合适的输入信号形式
    win_patch_rppg = pad.preprocess.standardize(win_rppg)

    y_x, e_x, _ = Method(win_patch_rppg[time_x - 1:], noise_x, length=time_x, **params_x)   # 滤除X轴运动信号
    y_xy, e_xy, _ = Method(e_x, noise_y, length=time_y, **params_y)                         # 滤除Y轴运动信号
    y_xyz, e_xyz, _ = Method(e_xy, noise_z, length=time_z, **params_z)                      # 滤除Z轴运动信号

    return e_x, e_xy, e_xyz