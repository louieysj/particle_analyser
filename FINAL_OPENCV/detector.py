import cv2
import numpy as np
import time


class Detector:
    def __init__(self):
        pass

    def detect(self, image, depth=None, thresh=None):
        if depth is None:
            return detect_noDepth(image)
        else:
            # return detect_noDepth(image)
            return detect_withDepth(image, depth, depth_thresh=thresh)


def detect_noDepth(src_img, thresh=127, mode_choice=cv2.RETR_LIST, method_choice=cv2.CHAIN_APPROX_SIMPLE):
    # 1. 清除背景纹理
    preprocessed_img = wipe_bg_morphology(src_img)
    # 2. 阈值化，得到二值图，随后开闭运算
    gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # 开运算和闭运算
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # 3. 找到轮廓
    contours, _ = cv2.findContours(closing, mode_choice, method_choice)
    return contours


def detect_withDepth(src_img, depth_data, depth_thresh):
    # todo 新思路：cv2.adaptiveThreshold和canny结合得到轮廓
    # depth_edge = cv2.threshold(depth_edge, 90, 255, cv2.THRESH_BINARY)[1]
    # depth_edge = cv2.adaptiveThreshold(depth_edge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
    # color_map = cv2.applyColorMap(depth_edge.astype(np.uint8), cv2.COLORMAP_JET)

    # src_grey = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    depth_normed = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # sobel_x = cv2.Sobel(depth_data, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(depth_data, cv2.CV_64F, 0, 1, ksize=3)
    # gradient = np.abs(np.add(sobel_x, sobel_y)).astype(np.uint8)

    front_mask = wipe_bg_depth(src_img, depth_data, depth_thresh)
    # 去除背景上的细小白点, 以及扩大区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    front_mask = cv2.morphologyEx(front_mask, cv2.MORPH_OPEN, kernel)
    # front_mask = cv2.dilate(front_mask, kernel, iterations=1)

    front_depth = cv2.bitwise_and(depth_normed, front_mask)

    # 对于每一块mask，分别提取对应区域的gradient, norm, 然后取反
    # tmp = np.zeros_like(front_mask)
    # num_labels, labels = cv2.connectedComponents(front_mask, connectivity=8, ltype=cv2.CV_32S)
    # for i in range(1, num_labels):
    #     component_mask = np.uint8(labels == i) * 255
    #     component_mask = cv2.dilate(component_mask, kernel, iterations=2)
    #     front_gradient = cv2.bitwise_and(gradient, component_mask)
    #     front_gradient = cv2.normalize(front_gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     front_gradient = cv2.dilate(front_gradient, kernel, iterations=1)
    #     mask = cv2.bitwise_and(255 - front_gradient, component_mask)
    #     tmp = cv2.bitwise_or(mask, tmp)
    #     tmp[front_mask==0] = 128

    # front_only = cv2.bitwise_and(depth_normed, tmp)
    front_only = cv2.bitwise_and(depth_normed, front_depth)
    contours, _ = cv2.findContours(front_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )
    return contours
    # from utils.utils import draw_contours
    # res = draw_contours(src_img, contours)

    # cv2.imshow('tmp', cv2.resize(res, None, fx=0.25, fy=0.25))
    # cv2.waitKey(0)
    # return contours, None


def wipe_bg_morphology(src_bgr_img):
    st = time.time()
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(src_bgr_img, cv2.COLOR_BGR2GRAY)

    # 模糊
    # blur = cv2.GaussianBlur(gray, (33, 33), 0)
    blur = cv2.medianBlur(gray, 77)
    # blur = low_pass_filter(gray)
    # cv2.imshow("blur", blur)
    # cv2.waitKey()

    # 自适应阈值二值化
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作
    st1 = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (43, 43))
    iter = 3
    closing = cv2.dilate(thresh, kernel, iterations=iter)
    closing = cv2.erode(closing, kernel, iterations=iter)
    # closing = thresh
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    # 连通组件标记
    # num, labels = cv2.connectedComponents(closing)

    # # 绘制边界框或掩模
    # for i in range(1, num):
    #     mask = (labels == i).astype('uint8')
    #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     rect = cv2.boundingRect(contours[0])
    #     cv2.rectangle(src_bgr_img, rect, (0, 255, 0), 2)

    closing[closing > 0] = 1
    # color_mask = np.array([closing * [255, 0, 0][i] for i in range(3)]).transpose(1, 2, 0).astype(np.uint8)
    # result = cv2.addWeighted(src_bgr_img, 0.7, color_mask, 0.3, 2)
    # 分离前景
    # front_only = cv2.bitwise_and(src_bgr_img, src_bgr_img, mask=closing)

    # 计算背景平均值
    mean = []
    src_img = cv2.medianBlur(src_bgr_img, 3)
    for c in cv2.split(src_img):
        c = c[closing == 0]
        mean.append(int(c.mean()))
    # 修改背景值
    channels = list(cv2.split(src_img))
    for i, c in enumerate(channels):
        c[closing == 0] = mean[i]
        channels[i] = c
    front_only = cv2.merge(channels)

    # 显示结果
    # result = utils.resize_img(result, width=1280)
    # cv2.imshow('result', result)
    # cv2.imshow('front_only', utils.resize_img(front_only, width=1280))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f'{st1 - st}/{time.time() - st}')
    return front_only


def wipe_bg_depth(src_bgr_img, depth_data, threshold=250):
    front_mask = depth_data.copy()
    front_mask[front_mask <= threshold] = 0
    front_mask[front_mask > 0] = 255
    front_mask = np.uint8(front_mask)

    # color_mask = np.array([front_mask * [255, 0, 0][i] for i in range(3)]).transpose((1, 2, 0)).astype(np.uint8)
    # result = cv2.addWeighted(src_bgr_img, 0.7, color_mask, 0.3, 2)
    # 分离前景
    # front_only = cv2.bitwise_and(src_bgr_img, src_bgr_img, mask=front_mask)

    return front_mask
