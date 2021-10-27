
import cv2
import numpy as np

#######   training part    ###############
samples = np.loadtxt('tmp/generalsamples.data', np.float32)
responses = np.loadtxt('tmp/generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

# model = cv2.KNearest()
model = cv2.ml.KNearest_create()
# model.train(samples,responses)
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

im = cv2.imread('img/1.png')
out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

list = []

for cnt in contours:
    if cv2.contourArea(cnt) > 5:
        [x, y, w, h] = cv2.boundingRect(cnt)
        print([x, y, w, h])
        if (h > 13 and h < 16) or (h > 3 and h < 7 and w > 3):
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = thresh[y:y+h, x:x+w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(
                roismall, k=1)
            string = str(int((results[0][0])))
            print(type(string))
            print(chr(int(string)))

            list.append([x, chr(int(string))])
            cv2.putText(out, chr(int(string)), (x, y+h), 0, 1, (0, 255, 0))

num = np.asarray(list)

data = num[num[:, 0].argsort()]  # 通过x轴排序
# data = data[:,data[2].argsort()]
print(data)
data = data[:, 1]
list = data.tolist()
string = ''.join(list)
print(string)  # 识别后的字符

cv2.imshow('im', im)
cv2.imshow('out', out)
cv2.waitKey(0)
