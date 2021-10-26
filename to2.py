import cv2  # 导入opencv库

# 读取一张图片，地址不能带中文
imgviewx = cv2.imread("img/1.png")

imgviewx2 = imgviewx.copy()
# 得到灰度图片
imgviewx2 = cv2.cvtColor(imgviewx2, cv2.COLOR_BGR2GRAY)
# 二值化图像，黑白图像，只有0和1,0为0,1为255
ret, imgviewx2 = cv2.threshold(
    imgviewx2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 二值化方法2
imgviewx2 = cv2.adaptiveThreshold(
    imgviewx2, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

cv2.imwrite("tmp/1-1.png", imgviewx2)
