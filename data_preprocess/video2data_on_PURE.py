import mediapipe as mp
import pyVHR as vhr
import sys
sys.path.append('..')
from SkinColorMagnification import extract 
from SkinColorMagnification.extract.utils import *
import numpy as np
import visualize_demo
from mediapipe.tasks import python  
from mediapipe.tasks.python import vision
import cv2
import math
import sys, os


def getRawRGBSignal_AND_ROI1(dataset_name, video_DIR, BVP_DIR, video_idx):

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
    visualize_demo.interactive_image_plot(visualize_skin_coll, 1.0)

    # hol_sig[N,1,3], N表示帧数, 1表示整张人脸, 3表示3个颜色通道；
    # visualize_skin_coll是一个存储包含人脸ROI图像的列表，其中元素为数组[H,W,3], H为每帧图像的高，W为每帧图像的宽，3表示3个颜色通道；
    # test_bvp[1,M]为参考信号
    return hol_sig, visualize_skin_coll, test_bvp                                                    # 函数返回多个值，自动封装为元组

def getRawRGBSignal_AND_ROI2(video_DIR):

    """
    跑自己拍摄的视频，提取的ROI是整张人脸区域
    :param video_DIR:视频存储的路径
    :return: 提取出的RGB信号、提取出的包含人脸ROI的视频帧序列
    """

    print('Video processed name: ', video_DIR)
    fps = extract.get_fps(video_DIR)
    print('Video frame rate: ', fps)
    # 播放视频
    # visualize_demo.display_video(video_DIR)
    # 设置皮肤提取方法
    sig_extractor = extract.SignalProcessing()
    sig_extractor.set_skin_extractor(extract.SkinExtractionConvexHull())
    seconds = 0
    sig_extractor.set_total_frames(seconds * fps)
    sig_extractor.set_visualize_skin_and_landmarks(                                                  # 设置可视化模式
        visualize_skin=True,
        visualize_landmarks=True,
        visualize_landmarks_number=True,
        visualize_patch=True)
    # Holistic提取，全部区域取均值
    hol_sig = sig_extractor.extract_holistic(video_DIR)
    visualize_skin_coll = sig_extractor.get_visualize_skin()
    # 播放提取的皮肤
    # visualize_demo.interactive_image_plot(visualize_skin_coll, 1.0)

    return hol_sig  # , visualize_skin_coll

def img_roi_mean(im):
    '''
        将一个ROI区域RGB图像的像素取平均得到一个像素值
    '''
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b
    return mean

def getRawRGBSignal_AND_ROI3(video_DIR):
    '''
        跑自己的视频，提取的ROI区域是视频中背景区域的像素
    '''
    print('Video processed name: ', video_DIR)
    fps = extract.get_fps(video_DIR)
    print('Video frame rate: ', fps)
    # 播放视频
    visualize_demo.display_video(video_DIR)
    # 读取第一帧,获取第一帧的背景区域ROI，之后的帧以此为标准
    sig = []
    vidcap = cv2.VideoCapture(video_DIR)
    ret, baseframe = vidcap.read()
    select_data = cv2.selectROI(baseframe)
    for frame in extract_frames_yield(video_DIR):
        img_roi = frame[int(select_data[1]):int(select_data[1] + select_data[3]), int(select_data[0]):int(select_data[0] + select_data[2])]
        # cv2.imshow("imageroi", img_roi)
        # cv2.waitKey(0)
        sig.append(img_roi_mean(img_roi))
        '''
        selectROI(windowName, img, showCrosshair=None, fromCenter=None):
            参数windowName: 选择的区域被显示在的窗口的名字
            参数img: 要在什么图片上选择ROI
            参数showCrosshair: 是否在矩形框里画十字线
            参数fromCenter: 是否是从矩形框的中心开始画
            返回值：一个元组[min_x, min_y, w, h]
                第一个值为矩形框中的最小x值
                第二个值为矩形框中的最小y值
                第三个值为这个矩形框的宽
                第四个值为这个矩形框的高
        '''
    sig = np.array(sig, dtype=np.float32)
    return sig

def getRawRGBSignal_AND_ROI4(video_DIR):
    '''
        跑自己的视频，提取的ROI区域是视频中头发区域/人肩膀区域的像素
        人是在动的，所以不能用第一帧为基准，与3的方法不一样
        涉及到目标检测，每帧的区域是不同的
        # 具体做法：(该做法被思考后未使用，因为相对位置可能在不同帧中也是不同的，因为可能被测对象相对于相机也有前后运动)
        #     虽然每帧头部位置是在发生变化的，但是头部两个点之间的相对位置是不变的；
        #     首先通过第一帧获取这种相对位置关系：例如检测人脸区域的一个特征点，手动选择头发区域的一块ROI，计算得到相对位置
        #     后续每一帧只需检测人脸区域的

        使用mediapipe框架中图像分割的算法
    '''
    print('Video processed name: ', video_DIR)
    fps = extract.get_fps(video_DIR)
    print('Video frame rate: ', fps)
    # 播放视频
    visualize_demo.display_video(video_DIR)

    sig = []

    # Height and width that will be used by the model
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    # Performs resizing and showing the image（输入模型的图片大小是固定的下，需要对齐进行修剪）
    def resize_and_show(image):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
        cv2.imshow("image", img)
        cv2.waitKey(0)

    BGb_COLOR = (0, 0, 0)  # black
    BG_COLOR = (192, 192, 192)  # gray
    MASK_COLOR = (255, 255, 255)  # white

    # Create the options that will be used for ImageSegmenter
    base_options = python.BaseOptions(model_asset_path='D:/PycharmProjects/pythonProject1/SkinColorMagnification/models/hair_segmenter.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True) # 如果设置为True，输出包括一个uint8格式图像的分割掩码，其中每个像素值表示获得的类别值

    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:

        # 循环浏览演示图像
        for frame in extract_frames_yield(video_DIR):
            # Create the MediaPipe image file that will be segmented
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # mp.Image是以几种格式之一来存储图像或者视频帧的一种容器。create_from_file是该容器的方法，指从图像文件中创建对象；此处没有文件则不需要用这个方法

            # 检索分割图像的掩码
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask

            # 生成纯色图像以显示输出分割掩码
            image_data = image.numpy_view()  # 调用该方法可以使得图像像素值作为numpy ndarray被检索；返回的numpy narray是对内部数据的引用，其本身是不可写的。如果调用者想要修改numpy narray，就要获取它的副本。
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BGb_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2  # 创建条件矩阵，将原始掩码数组的三个副本沿最后一个轴排列
            output_image = np.where(condition, image_data, bg_image)  # 根据条件矩阵来选择前景图像或者背景图像的相应像素值，生成一个新的输出图像；
                                                                    # 在对应位置上进行判断。如果条件矩阵中的元素为 True，则对应位置上的 output_image 的像素值将使用 fg_image 中
                                                                    # 的像素值；如果条件矩阵中的元素为 False，则对应位置上的 output_image 的像素值将使用 bg_image 中的像素值。
            sig.append(img_roi_mean(output_image))
            # resize_and_show(output_image)

    sig = np.array(sig, dtype=np.float32)
    return sig

if __name__ == "__main__":
    dataset_name = 'pure'
    video_DIR = 'D:/database/PURE/'
    BVP_DIR = 'D:/database/PURE/'
    # 批量跑数据
    for i in range(58):
                raw_RGB_signal, raw_ROI, test_bvp = getRawRGBSignal_AND_ROI1(dataset_name, video_DIR, BVP_DIR, i)
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

