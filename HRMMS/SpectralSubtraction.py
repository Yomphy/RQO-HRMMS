import scipy
import numpy as np
from AMTC import *

# 运动信号时频图做掩模与rPPG信号时频图相乘，直接置零
# def Spectrum_sub(rppgs, speedx, speedy, speedz, stride, fps, wsize):
#     win = 0
#     pujian_rppgs = []
#     for win_rppgs in rppgs:
#         patch = 0
#         patch_rppgs = []
#         for patch_rppg in win_rppgs:
#             spx = speedx[int(win * stride * fps): int(win * stride * fps + wsize * fps)]
#             spy = speedy[int(win * stride * fps): int(win * stride * fps + wsize * fps)]
#             spz = speedz[int(win * stride * fps): int(win * stride * fps + wsize * fps)]
#             _, _, spec_x = scipy.signal.stft(spx, fs=fps, nperseg=200, noverlap=196, nfft=1024)
#             _, _, spec_y = scipy.signal.stft(spy, fs=fps, nperseg=200, noverlap=196, nfft=1024)
#             _, _, spec_z = scipy.signal.stft(spz, fs=fps, nperseg=200, noverlap=196, nfft=1024)
#             spec_x = np.abs(spec_x) / np.max(np.max(np.abs(spec_x)))
#             spec_y = np.abs(spec_y) / np.max(np.max(np.abs(spec_y)))
#             spec_z = np.abs(spec_z) / np.max(np.max(np.abs(spec_z)))
#             rppg_f, rppg_t, spec_rppg = scipy.signal.stft(patch_rppg, fs=fps, nperseg=200, noverlap=196, nfft=1024)
#
#             spec_x = np.where(spec_x > 0.2, 0, 1)
#             spec_y = np.where(spec_y > 0.2, 0, 1)
#             spec_z = np.where(spec_z > 0.2, 0, 1)
#             spec_mask = np.multiply(np.multiply(spec_x, spec_y), spec_z)
#             spec_rppg = np.multiply(spec_rppg, spec_mask)
#             _, rppg_istft = scipy.signal.istft(spec_rppg, fs=fps, nperseg=200, noverlap=196, nfft=1024)
#
#             patch_rppgs.append(np.array(rppg_istft))
#             patch += 1
#         win += 1
#         pujian_rppgs.append(np.array(patch_rppgs))
#
#     return pujian_rppgs


# 结合了AMTC寻迹，运动轨迹衰减的谱减法
def Spectrum_sub(rppgs, speedx, speedy, speedz, stride, fps, wsize):
    win = 0
    pujian_rppgs = []
    for win_rppgs in rppgs:
        patch = 0
        patch_rppgs = []
        for patch_rppg in win_rppgs:
            spx = speedx[int(win * stride * fps): int(win * stride * fps + wsize * fps)]
            spy = speedy[int(win * stride * fps): int(win * stride * fps + wsize * fps)]
            spz = speedz[int(win * stride * fps): int(win * stride * fps + wsize * fps)]
            fx, tx, spec_x = scipy.signal.stft(spx, fs=fps, nperseg=200, noverlap=196, nfft=1024)
            fy, ty, spec_y = scipy.signal.stft(spy, fs=fps, nperseg=200, noverlap=196, nfft=1024)
            fz, tz, spec_z = scipy.signal.stft(spz, fs=fps, nperseg=200, noverlap=196, nfft=1024)
            spec_x = np.abs(spec_x) / np.max(np.max(np.abs(spec_x)))
            spec_y = np.abs(spec_y) / np.max(np.max(np.abs(spec_y)))
            spec_z = np.abs(spec_z) / np.max(np.max(np.abs(spec_z)))
            spec_xyz = spec_x + spec_y + spec_z
            rppg_f, rppg_t, spec_rppg = scipy.signal.stft(patch_rppg, fs=fps, nperseg=200, noverlap=196, nfft=1024)

            f_length = int(300 / 60 / rppg_f[1])

            speed_tracker = AMTC()
            speed_tracker.show_groundtruth = False
            speed_tracker.show_traces = False
            speed_tracker.set_parameters(spec_xyz[:f_length, :], fx[:f_length], tx, num=2, threshold=0.05, ground_truth=None,
                                         scale=1)
            _, _, _ = speed_tracker.run_AMTC(lam=0.1)
            speed_roads = speed_tracker.get_roads()
            energy_xyz = spec_x + spec_y + spec_z
            for road in speed_roads:
                road = road_hampel_filter(road)
                road = trace_exited(rppg_f[:f_length], np.abs(spec_rppg[:f_length, :]), road)
                spec_rppg = damping_energy(rppg_f[:f_length], spec_rppg[:f_length, :], road, energy_xyz[:f_length, :])

            _, rppg_istft = scipy.signal.istft(spec_rppg, fs=fps, nperseg=200, noverlap=196, nfft=1024)

            patch_rppgs.append(np.array(rppg_istft))
            patch += 1
        win += 1
        pujian_rppgs.append(np.array(patch_rppgs))

    return pujian_rppgs


# 每个窗信号会进行动作分类，窗信号进行不同的滤波选择
def spectrum_sub_motion_class(rppg, noise_spec, fps, nperseg, overlap, nfft, strong=10, road_list=None, trace_existed=True):
    ff, tt, spec_rppg = scipy.signal.stft(rppg, fs=fps, nperseg=nperseg, noverlap=overlap, nfft=nfft)     # rPPG信号的时频图
    f_length = int(240 / 60 / ff[1])   # 只在0-4Hz范围内进行能量衰减

    speed_tracker = AMTC()            # AMTC频率追踪
    speed_tracker.show_groundtruth = False
    speed_tracker.show_traces = False
    speed_tracker.set_parameters(noise_spec[:f_length, :], ff[:f_length], tt, num=1, threshold=0.05, ground_truth=None, scale=1)  # 参数设置，再速度信号的时频图上追踪运动信号的频率轨迹
    _, _, _ = speed_tracker.run_AMTC(lam=5)   # 开始运行
    road = speed_tracker.get_roads()[-1]    # 获取追踪结果，返回的是一个列表，列表长度等于设置的追踪个数num，每个元素是频率路径索引
    road_mean = np.mean(road)

    roads = []
    if road_list is None:
        road_list = [1, 2]
    else:
        pass
    for i in road_list:       # 根据基频轨迹计算二倍频谐波轨迹
        roads.append([index * i if index * i < f_length else index // 2 for index in road])   # 防止轨迹频率溢出

    noise_xyz = noise_spec
    for road in roads:    # 对每一条频率轨迹的能量进行衰减
        road = hampel_filter(road)   # 首先奇异值滤波，排除奇异值
        if trace_existed:          # 检验该条轨迹在rPPG信号中是否存在，轨迹能量阈值判断
            road = trace_exited(ff[:f_length], np.abs(spec_rppg[:f_length, :]), road)

        spec_rppg = damping_energy(ff[:f_length], spec_rppg[:f_length, :], road, noise_xyz[:f_length, :], strong)  # 根据速度信号的轨迹在rPPG信号上对其进行能量衰减。

    _, rppg_istft = scipy.signal.istft(spec_rppg, fs=fps, nperseg=nperseg, noverlap=overlap, nfft=nfft)  # 能量衰减后的rPPG信号时频图信息和保留的相位信息，进行ISTFT得到时域信号并返回。
    return rppg_istft  # 返回降噪后rPPG信号


# def spectrum_sub_motion_class(rppg, noise_spec, fps, nperseg, overlap, nfft):
#     spec_noise = noise_spec / np.max(np.max(noise_spec))
#
#     rppg_f, rppg_t, spec_rppg = scipy.signal.stft(rppg, fs=fps, nperseg=nperseg, noverlap=overlap, nfft=nfft)
#     power_noise = np.abs((np.where(spec_noise > 0.1, 1, 0) * spec_rppg)) ** 2
#     power_rppg = np.abs(spec_rppg) ** 2
#     power_sub = power_rppg - power_noise
#
#     phase_rppg = np.angle(spec_rppg)
#     mag_rppg = np.sqrt(power_sub)
#     new_rppg = mag_rppg * np.exp(1j * phase_rppg)
#     _, rppg_istft = scipy.signal.istft(new_rppg, fs=fps, nperseg=200, noverlap=196, nfft=1024)
#
#     return rppg_istft
