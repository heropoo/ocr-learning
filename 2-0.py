import cv2
import numpy as np
from util import *

'''
载入图像
灰度化
二值化
轮廓检测
@see https://blog.csdn.net/zb1165048017/article/details/109404373
'''
# 载入图像
img = cv2.imread("img/test/1.png", cv2.IMREAD_UNCHANGED)

# 灰度化
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
ret, bin_img = cv2.threshold(
    gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# showImg(bin_img, 'bin_img')

# bin_img1 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
#             cv2.THRESH_BINARY,11,2)
# showImg(bin_img1, 'bin_img1')

# bin_img2 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,11,2)
# showImg(bin_img2, 'bin_img2')

# 轮廓检测
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img,contours,-1,(0,255,0),5)

# print(contours)
# print(hierarchy)

max_area = 0.0
max_perimeter = 0.0
max_index1 = 0
max_index2 = 0

for i in range(len(contours)):
    cnt = contours[i]

    # M = cv2.moments(cnt)

    # 获取指定轮廓所包含的面积
    area = cv2.contourArea(cnt)
    # print(area)

    # 获取指定轮廓所包含的周长，第二个参数指示当前输入为闭合轮廓(true)还是非闭合曲线(false)
    perimeter  = cv2.arcLength(cnt, True)
    # print(perimeter)

    if(area > max_area):
        max_area = area
        max_index1 = i

    if(perimeter > max_perimeter):
        max_perimeter = perimeter
        max_index2 = i

print(max_area, max_index1, max_perimeter, max_index2)

# 只画最大的轮廓
cv2.drawContours(img,contours,max_index1,(0,255,0),5)

showImg(img, 'img')



