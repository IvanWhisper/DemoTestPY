#coding=utf-8
import cv2
# import cv2.cv as cv

img = cv2.imread(r".\testpic\test6.jpg")

def detect(img, cascade):
    '''detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，
faces表示检测到的人脸目标序列，1.3表示每次图像尺寸减小的比例为1.3，
 4表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
 CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(20, 20)为目标的最小最大尺寸'''
    rects = cascade.detectMultiScale(img)
                                     # , scaleFactor=1.3,
                                     # minNeighbors=5, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    print(rects)
    return rects

#在img上绘制矩形
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


#转换为灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#直方图均衡处理
gray = cv2.equalizeHist(gray)

#脸部特征分类地址，里面还有其他
cascade_fn = r".\data\haarcascades_cuda\haarcascade_frontalface_alt.xml"

#读取分类器,CascadeClassifier下面有一个detectMultiScale方法来得到矩形
cascade = cv2.CascadeClassifier(cascade_fn)

#通过分类器得到rects
rects = detect(gray, cascade)
while True:
    #vis为img副本
    vis = img.copy()
    #画矩形
    draw_rects(vis, rects, (0, 255, 0))
    n=0
    for x1, y1, x2, y2 in rects:
        #print(x1,y1,x2,y2)
        #   roi = gray[y1:y2, x1:x2]
        #   vis_roi = vis[y1:y2, x1:x2]
        # print(x1,y1,x2,y2)
        crop = vis[y1:y2, x1:x2]
        #cv2.imshow('crop', crop)
        cv2.imwrite(r".\PickFaceImg\imgpick"+str(n)+".jpg", crop)
        n+=1
        #  subrects = detect(roi.copy(), nested)
        #  draw_rects(vis_roi, subrects, (255, 0, 0))
    # draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
    cv2.imshow('facedetect', vis)
    cv2.imwrite(r".\LocationImg\imglocal.jpg",vis)

    #print(rects.count())
    print(len(rects))

    if n>len(rects)-1:
        break

exit()