import cv2
import open3d as o3d
import numpy as np
import time

# img_path = r'Shuojia Yan/3.15/demo1.tif'
# depth_path = r'Shuojia Yan\3.15\demo13dview.csv'
img_path = r'3d data/6.jpg'
depth_path = r'3d data/6.csv'
# img_path = r'Shuojia Yan/newdemo.jpg'
# depth_path = r'Shuojia Yan/newdemo.csv'

image = cv2.imread(img_path)
color_image = o3d.geometry.Image(image)
depth_image = np.loadtxt(depth_path, delimiter=',').astype(np.int16)
depth_image[depth_image == -1] = 0


def detect_contours(src_img, thresh=127, mode_choice=cv2.RETR_TREE, method_choice=cv2.CHAIN_APPROX_NONE):
    st = time.time()
    historys = {}
    # 1. 清除背景纹理
    preprocessed_img = wipe_bg(src_img)
    st1 = time.time()
    historys['preprocess'] = preprocessed_img
    # 2. 阈值化，得到二值图，随后开闭运算
    gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # 开运算和闭运算
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    historys['morphologyEx'] = closing
    # 3. 找到轮廓
    contours, _ = cv2.findContours(closing, mode_choice, method_choice)
    print(f'{st1 - st}/{time.time() - st}')
    return contours, historys


def wipe_bg(src_bgr_img, depth_img):
    """
    清除大部分的背景纹理
    """

    front_mask = depth_img.copy()
    thresh = 100
    front_mask[front_mask <= thresh] = 0
    front_mask[front_mask > 0] = 1
    front_mask = np.uint8(front_mask)

    color_mask = np.array([front_mask * [255, 0, 0][i] for i in range(3)]).transpose(1, 2, 0).astype(np.uint8)
    result = cv2.addWeighted(src_bgr_img, 0.7, color_mask, 0.3, 2)
    # 分离前景
    front_only = cv2.bitwise_and(src_bgr_img, src_bgr_img, mask=front_mask)

    return front_only, front_mask

    # 模糊
    # blur = cv2.GaussianBlur(gray, (33, 33), 0)
    blur = cv2.medianBlur(gray, 77)
    # blur = low_pass_filter(gray)
    # cv2.imshow("blur", blur)
    # cv2.waitKey()

    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作
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
    color_mask = np.array([closing * [255, 0, 0][i] for i in range(3)]).transpose(1, 2, 0).astype(np.uint8)
    result = cv2.addWeighted(src_bgr_img, 0.7, color_mask, 0.3, 2)
    # 分离前景
    front_only = cv2.bitwise_and(src_bgr_img, src_bgr_img, mask=closing)

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
    return front_only


front_img, front_mask = wipe_bg(image, depth_image)

exit(0)

depth_image = depth_image.max() - depth_image
depth_image = o3d.geometry.Image(depth_image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=color_image,
    depth=depth_image,
    depth_scale=1e2,
    depth_trunc=100.0,
    convert_rgb_to_intensity=False
)
color_raw = o3d.io.read_image("test_data/RGBD/color/00000.jpg")
depth_raw = o3d.io.read_image("test_data/RGBD/depth/00000.png")
color_raw = o3d.geometry.Image(cv2.imread("test_data/RGBD/color/00000.jpg"))
# depth_raw = o3d.io.read_image("test_data/RGBD/depth/00000.png")
rgbd_image_redwood = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, convert_rgb_to_intensity=False)

f = 1e4  # 相机焦距（单位：微米）
p = 2.171994448  # 像素尺寸（单位：微米）(比例尺)
focal_length = f / p
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=image.shape[1], height=image.shape[0], fx=focal_length, fy=focal_length, cx=image.shape[1] / 2,
    cy=image.shape[0] / 2
),
intrinsic.set
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    image=rgbd_image,
    intrinsic=intrinsic,
    # intrinsic=o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    # )
    # extrinsic=np.eye(4),
)
# 可视化点云
o3d.visualization.draw_geometries([pcd])

exit(0)

# 读取深度数据
image = cv2.imread(img_path)
depth_map = np.loadtxt(depth_path, delimiter=',')

normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
color_map = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

result = cv2.addWeighted(image, 0.8, color_map, 0.2, 0)

# 计算水平方向上的梯度
sobelx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)

# 计算垂直方向上的梯度
sobely = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

result = np.add(sobelx, sobely)
result = np.abs(result)
# _, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)
normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
color_map = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
result = cv2.addWeighted(image, 0.8, color_map, 0.2, 0)

# 将计算出的梯度值合并为一幅梯度图像
sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
sobel_mag = np.uint8(sobel_mag)

_, thresh = cv2.threshold(sobel_mag, 100, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.erode(thresh, kernel, iterations=1)
thresh = cv2.dilate(thresh, kernel, iterations=1)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Sobel Magnitude", sobel_mag)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
