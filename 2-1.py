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
img_show = img.copy()
cnt_idx = 184
cnt = contours[cnt_idx]

x,y,w,h = cv2.boundingRect(cnt)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

#无旋转矩形
cv2.rectangle(img_show,(x,y),(x+w,y+h),(0,255,0),4)
#有旋转矩形
cv2.drawContours(img_show,[box],0,(0,0,255),4)

showImg(img_show, 'img_show')



