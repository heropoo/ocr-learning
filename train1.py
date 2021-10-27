import sys
import numpy as np
import cv2

im = cv2.imread('img/1.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100))
responses = []
keys = [i for i in range(44, 58)]

for cnt in contours:
    if cv2.contourArea(cnt) > 5:  # 大于像素点的区域
        [x, y, w, h] = cv2.boundingRect(cnt)
        print(x, y, w, h)
        continue
        print([x, y, w, h])  # 对应的区域的坐标
        if (h > 13 and h < 20) or (h > 2 and h < 7 and w > 4):  # 筛选不需要的区域
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi = thresh[y:y+h, x:x+w]
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)
            print(key)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                key = str(key)
                print(key)
                responses.append(int(key))  # 保存ascii码
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)

# responses = np.array(responses, np.float32)
# responses = responses.reshape((responses.size, 1))
# print("training complete")
# print(samples)
# print(responses)
# #
# np.savetxt('tmp/generalsamples.data', samples)
# np.savetxt('tmp/generalresponses.data', responses)
