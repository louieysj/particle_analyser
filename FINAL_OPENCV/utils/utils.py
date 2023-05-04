import time
from PIL import Image
import cv2
import math
import numpy as np
import os

colors = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (67, 99, 216),
    (245, 130, 48),
    (145, 30, 180),
    (66, 212, 244),
    (240, 50, 230),
    (191, 239, 69),
    (250, 190, 212),
    (70, 153, 144),
    (220, 190, 255),
    (154, 99, 36),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 216, 177),
    (0, 0, 117)
]


def resize_img(src, scale=None, w=None, h=None):
    # 如果指定了scale，则无视w,h
    # 如果只指定了w, h其一，则按比例缩放
    # 否则按照w, h大小resize
    if scale:
        return cv2.resize(src, None, fx=scale, fy=scale)
    elif w is not None and h is not None:
        return cv2.resize(src, (w, h))
    else:
        if w:
            new_w = int(w)
            new_h = int(w * src.shape[0] / src.shape[1])
        else:
            new_h = int(h)
            new_w = int(h * src.shape[1] / src.shape[0])
        return cv2.resize(src, (new_w, new_h))


def scale_contour_ToShow(contours, scale):
    contours = list(contours)
    for i, c in enumerate(contours):
        c = c * scale
        contours[i] = np.int32(c)
    return contours


def findNearestContour(contours, x, y):
    min_dist = np.inf
    min_idx = -1

    # 遍历所有轮廓，计算与目标坐标的距离，找到距离最小的轮廓
    for i, contour in enumerate(contours):
        dist = cv2.pointPolygonTest(contour, (x, y), True)
        if abs(dist) < abs(min_dist):
            min_dist = dist
            min_idx = i
    return min_idx, min_dist


def findBoundingContours(contours, x, y):
    # 找到包围这个点的所有轮廓
    point = (x, y)
    bounding_contours = []
    for idx, contour in enumerate(contours):
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            bounding_contours.append(idx)
    return bounding_contours


def findNearestPoint_on_contour(x, y, contour):
    # 遍历所有轮廓
    min_dist = float('inf')
    for i, pt in enumerate(contour):
        # 计算当前点到待查询点的距离
        dist = np.linalg.norm(pt - (x, y))

        # 如果当前距离比之前的距离更近，更新最近点和距离
        if dist < min_dist:
            nearest_pt = list(pt[0])
            min_dist = dist
    # print(f'nearest point:{nearest_pt}')
    return i, nearest_pt


def split_contour(contours, nearest_contour_idx, tmp_contour):
    contour = contours[nearest_contour_idx]
    contour = [list(i[0]) for i in contour]
    idx1 = contour.index(tmp_contour[0])
    idx2 = contour.index(tmp_contour[-1])
    if idx1 < idx2:
        c1 = contour[idx1:idx2 + 1]
        c2 = contour[idx2:] + contour[:idx1]
    else:
        c2 = contour[idx2:idx1 + 1]
        c1 = contour[idx1:] + contour[:idx2]
    c1 += tmp_contour[::-1]
    c2 += tmp_contour[::]
    c1 = np.array(c1).reshape(len(c1), 1, 2)
    c2 = np.array(c2).reshape(len(c2), 1, 2)

    del contours[nearest_contour_idx]
    contours.append(c1)
    contours.append(c2)
    return contours


def DeleteEdgeContour(contours, w, h):
    # 删除边界轮廓
    delete_idx = []
    for idx, contour in enumerate(contours):
        cnt = 0
        for p in contour:
            x, y = p[0]
            if x in (0, w - 1) or y in (0, h - 1):
                cnt += 1
                if cnt >= 2:
                    delete_idx.append(idx)
                    break
    for idx in delete_idx[::-1]:
        del contours[idx]

    return contours


def draw_contours(img, contours, line_width=10, selected_idx=-1):
    img = img.copy()
    if not contours:
        return
    # 遍历所有轮廓并在图像上绘制它们
    for i, c in enumerate(contours):
        # cv2.drawContours(image, [c], -1, (36,255,12), 2)
        color = None
        if i == selected_idx:
            color = (144, 238, 144)
        else:
            color = colors[i % len(colors)]
        cv2.drawContours(img, [c], -1, color, line_width if i != selected_idx else line_width * 3)
    return img


def read_depth_from_csv(csv_path):
    # st = time.time()
    folder, filename = os.path.split(csv_path)
    int16_path = os.path.join(folder, 'PREPROCESSED_' + os.path.splitext(filename)[0] + '.png')
    if os.path.exists(int16_path):
        data = np.array(Image.open(int16_path))
    else:
        data = np.loadtxt(csv_path, delimiter=',')
        data = replace_minus1(data)
    # st1 = time.time()

    # print(f'{st1-st}/{time.time()-st}')
    return data


def replace_minus1(data):
    # 把-1的值转成四周的平均值
    coordinates = np.argwhere(data == -1).tolist()
    center = np.array(data.shape[:2]) / 2
    # 对 coordinates 列表进行排序，根据与中心点的距离从小到大排序
    sorted_coordinates = list(sorted(coordinates, key=lambda c: np.linalg.norm(np.array(c) - center)))
    kernel_size = 3
    kz = int((kernel_size - 1) / 2)
    while sorted_coordinates:
        idx = 0
        while idx < len(sorted_coordinates):
            c = sorted_coordinates[idx]
            i, j = c
            submatrix = data[max(0, i - kz):min(i + kz + 1, data.shape[0]),
                        max(0, j - kz):min(j + kz + 1, data.shape[1])]
            if np.count_nonzero(submatrix != -1):  # 不是全为-1
                data[i, j] = np.mean(submatrix[submatrix != -1])
                sorted_coordinates.remove(c)
            else:
                idx += 1
    return data


def cal_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    return circularity


def cal_diameter(contour):
    # 计算直径
    x, y, w, h = cv2.boundingRect(contour)
    d = math.sqrt(w * w + h * h) / 2
    return d


def cal_area(contour):
    return cv2.contourArea(contour)