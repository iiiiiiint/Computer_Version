import cv2
import numpy as np
from tqdm import tqdm

# 步骤1: 相机标定
# 标定板的格子数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
pattern_size = (9, 6)
# 创建一个数组用于保存标定板角点的坐标
objpoints = []
imgpoints = []

# 生成标定板上的三维点
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# 读取标定板图片
calibration_images = ['./chessboard/IMG_5877.JPG', './chessboard/IMG_5878.JPG', './chessboard/IMG_5879.JPG', './chessboard/IMG_5880.JPG', './chessboard/IMG_5882.JPG',
          './chessboard/IMG_5883.JPG', './chessboard/IMG_5884.JPG', './chessboard/IMG_5885.JPG', './chessboard/IMG_5887.JPG', './chessboard/IMG_5888.JPG',
          './chessboard/IMG_5890.JPG', './chessboard/IMG_5893.JPG', './chessboard/IMG_5894.JPG', './chessboard/IMG_5895.JPG', './chessboard/IMG_5896.JPG',
          './chessboard/IMG_5897.JPG', './chessboard/IMG_5898.JPG', './chessboard/IMG_5899.JPG']

for fname in tqdm(calibration_images):
    img = cv2.imread(fname)
    img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找标定板角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    # print(corners.shape)
    if ret:
        # 如果找到角点，添加对象点和图像点
        # cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("内参数矩阵...")
print(mtx)
print("畸变系数...")
print(dist)

# 步骤2: 获取新的相机矩阵
img = cv2.imread('IMG_5877.JPG')
img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

h, w = img.shape[:2]


new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 步骤3: 使用新的相机矩阵进行畸变矫正
undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

ret, corners = cv2.findChessboardCorners(undistorted_img, pattern_size, None)
if ret:
    # 定义转换后的图像中的四个角点（鸟瞰视角）
    src_points = np.float32([corners[0], corners[pattern_size[0] - 1], corners[-1], corners[-pattern_size[0]]])
    dst_points = np.float32([[200, 200], [w/3+200, 200], [w/3+200, h/3+200], [200, h/3+200]])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 应用透视变换
    birdseye_img = cv2.warpPerspective(undistorted_img, M, (w, h), flags=cv2.INTER_LINEAR)

    # 显示结果
    cv2.imwrite('UndistortedImage.jpg', undistorted_img)
    cv2.imwrite('BirdseyeView.jpg', birdseye_img)

else:
    print("未找到标定板角点。")
