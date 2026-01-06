import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, fs, cutoff=0.6, order=5):
    """
    5阶巴特沃斯低通滤波器，截止频率默认0.6Hz

    Args:
        signal (np.ndarray): 输入信号
        fs (float): 采样率
        cutoff (float): 截止频率（Hz）
        order (int): 滤波器阶数

    Returns:
        np.ndarray: 滤波后的信号
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered


def rgb_to_yuv_batch(rgb_data):
    assert rgb_data.shape[1] == 3
    r, g, b = rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return np.stack([y, u, v], axis=1)

def RQO(RGB, fs):
    YUV = rgb_to_yuv_batch(RGB)

    def shift_signal(signal, T):
        return np.roll(signal, -T)

    X = np.vstack([
        YUV[:,1] - lowpass_filter(YUV[:,1], fs),
        YUV[:,2] - lowpass_filter(YUV[:,2], fs)
    ]).T

    def global_optimization(X, min_period, max_period):
        best_w, best_T, min_cost = None, None, float('inf')
        for T in range(min_period, max_period):
            shifted_X = np.vstack([shift_signal(X[:,0], T), shift_signal(X[:,1], T)]).T
            diff_X = X - shifted_X
            A = diff_X.T @ diff_X
            eigenvalues, eigenvectors = np.linalg.eig(A)
            w = eigenvectors[:, np.argmin(eigenvalues)]
            y = X @ w
            shifted_y = shift_signal(y, T)
            cost = np.linalg.norm(y - shifted_y)**2
            if cost < min_cost:
                min_cost, best_w, best_T = cost, w, T
        return best_w, best_T, min_cost

    best_w, best_T, _ = global_optimization(X, min_period=10, max_period=60)
    if best_w is not None:
        return (X @ best_w).flatten()
    else:
        raise ValueError("Optimization failed to find a valid solution.")
