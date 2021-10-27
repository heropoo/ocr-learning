import cv2  # 导入opencv库

def showImg(img, title="img"):
    cv2.imshow(title, img)  # 建立image窗口显示图片
    k = cv2.waitKey(0)  # 无限期等待输入

    if k == 27:  # 如果输入ESC退出
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('tmp/'+title+'.png', img)
        print("Save img OK!")
        cv2.destroyAllWindows()
