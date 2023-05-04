import open3d as o3d
import numpy as np
import cv2
from scipy.spatial import ConvexHull

# img_path = r'Shuojia Yan/3.15/demo1.tif'
# depth_path = r'Shuojia Yan\3.15\demo13dview.csv'
img_path = r'3d data/2.jpg'
depth_path = r'3d data/2.csv'
# img_path = r'Shuojia Yan/newdemo.jpg'
# depth_path = r'Shuojia Yan/newdemo.csv'


def read_pcd_from_img_and_csv(img_path, csv_path):
    image = cv2.imread(img_path)
    h, w, c = image.shape
    depth_image = np.loadtxt(csv_path, delimiter=',').astype(np.int16)
    depth_image[depth_image == -1] = 0

    scale = 0.5
    h = int(h*scale)
    w = int(w*scale)
    image = cv2.resize(image, (w, h))
    depth_image = cv2.resize(depth_image, (w, h))
    # 创建平面点云
    points_3d = np.zeros((h * w, 3))
    coords = np.indices((h, w)).transpose(1, 2, 0).reshape((-1, 2))
    points_3d[:, :2] = coords
    points_3d[:, -1] = depth_image.flatten() * scale

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # 设置点云颜色
    color = image.reshape((-1, 3)) / 255
    pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd


pcd = read_pcd_from_img_and_csv(img_path, depth_path)

import matplotlib.pyplot as plt
points = np.asarray(pcd.points)

print("->正在计算点云凸包...")
hull, _ = pcd.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcd, hull_ls])


o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# 创建Open3D可视化器对象
vis = o3d.visualization.Visualizer()
vis.create_window()

# 将点云添加到可视化器中
vis.add_geometry(pcd)

# 添加参考坐标系
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2000, origin=[0, 0, 0])
vis.add_geometry(coord_frame)

# 调整相机姿态
ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])
ctr.set_front([-1, -1, -1])
ctr.set_up([0, 0, 1])

# 显示点云
vis.run()
