import mediapipe as mp
import pyVHR as vhr
import sys
sys.path.append('..')
from pyVHR import extraction as extract
import numpy as np
import sys


def getRawRGBSignal(dataset_name, video_DIR, BVP_DIR, video_idx):

    """
    跑公开数据库的视频，提取的ROI是整张人脸区域
    :param dataset_name: 需要提取数据库的名称
    :param video_DIR: 存储视频的文件夹路径
    :param BVP_DIR: 存储参考信号的文件夹路径
    :param video_idx: 视频的序号
    :return: 提取出的RGB信号、提取出的包含人脸ROI的视频帧序列、参考信号
    """

    dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)  
    allvideo = dataset.videoFilenames                                                                
    for v in range(len(allvideo)):
        print(v, allvideo[v])

    fname = dataset.getSigFilename(video_idx)                                                        
    sigGT = dataset.readSigfile(fname)                                                             
    print('BVP processed name: ', fname)
    test_bvp = sigGT.data
  


    videoFileName = dataset.getVideoFilename(video_idx)
    print('Video processed name: ', videoFileName)
    fps = extract.get_fps(videoFileName)
    print('Video frame rate:     ', fps)
    # 播放视频
    visualize_demo.display_video(videoFileName)
    # 设置皮肤提取方法
    sig_extractor = extract.SignalProcessing()
    sig_extractor.set_skin_extractor(extract.SkinExtractionConvexHull())                             # 凸包法
    seconds = 0                                                                                      # 设置要处理的视频的长度（0对应于全部视频）
    sig_extractor.set_total_frames(seconds * fps)
    sig_extractor.set_visualize_skin_and_landmarks(                                                  # 设置可视化模式
        visualize_skin=True,
        visualize_landmarks=True,
        visualize_landmarks_number=True,
        visualize_patch=True)
    # Holistic提取，全部区域取均值
    hol_sig = sig_extractor.extract_holistic(videoFileName)                                          # 未经过放大的RGB信号
    visualize_skin_coll = sig_extractor.get_visualize_skin()                                         # 为一个列表，存放每帧裁剪的皮肤RGB图像
    # 播放提取的皮肤

    # hol_sig[N,1,3], N表示帧数, 1表示整张人脸, 3表示3个颜色通道；
    # visualize_skin_coll是一个存储包含人脸ROI图像的列表，其中元素为数组[H,W,3], H为每帧图像的高，W为每帧图像的宽，3表示3个颜色通道；
    # test_bvp[1,M]为参考信号
    return hol_sig, visualize_skin_coll, test_bvp                                                    # 函数返回多个值，自动封装为元组


if __name__ == "__main__":
    dataset_name = 'pure'
    video_DIR = 'D:/database/PURE/'
    BVP_DIR = 'D:/database/PURE/'
    # 批量跑数据
    for i in range(58):
                raw_RGB_signal, raw_ROI, test_bvp = getRawRGBSignal(dataset_name, video_DIR, BVP_DIR, i)
                # np.save('D:/database/NLG-rPPG/low'+str(i)+'rawRGBSignal.npy', raw_RGB_signal)
                # np.save('D:/database/NLG-rPPG/low'+str(i)+'rawPPG.npy', test_bvp)

    # raw_RGB_signal, raw_ROI = getRawRGBSignal_AND_ROI2('C:/Users/浮夸/Desktop/video1.avi')

    # p = 'D:/database/PURE/10-04'
    # scandir方法 返回一个DirEntry迭代器对象，并能告诉你迭代文件的路径，是一种目录迭代方法
    # 该方法具有一些属性和方法：name - 条目的文件名
    #                       path - 输入路径NAME(不一定是绝对路径)
    # for f in os.scandir(p):
    # f = os.scandir(p)
    # dir = p + '/10-04.avi'
    # print(dir)
    # raw_RGB_signal = getRawRGBSignal_AND_ROI4(dir)
    # np.save(p+'/rgb_hairsig.npy', raw_RGB_signal)

