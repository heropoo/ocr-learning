import cv2  # 导入opencv库

from util import *
import os

# current_path = os.getcwd()

# 读取一张图片，地址不能带中文
img = cv2.imread("img/test/1.png", cv2.IMREAD_UNCHANGED)
# print(img)

showImg(img, 'img')

# 复制图像
img2 = img.copy()

# 得到灰度图片
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

showImg(img2, 'img2')

# 二值化图像，黑白图像，只有0和1,0为0,1为255
ret, img2x = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print("二值化图像1: ", ret)

showImg(img2x, 'img2x')

# 二值化方法2
img2x = cv2.adaptiveThreshold(
    img2, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
print("二值化图像2: ")

showImg(img2x, 'img2xx')

#cv2.imwrite("tmp/1-1.png", imgviewx2)
