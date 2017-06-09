# -*- coding: utf-8 -*-
# feimengjuan
import numpy as np
import cv2


# 1、整个条形码的算法流程如下：
# 2、计算x方向和y方向上的Scharr梯度幅值表示
# 3、将x-gradient减去y-gradient来显示条形码区域
# 4、模糊并二值化图像
# 5、对二值化图像应用闭运算内核
# 6、进行系列的腐蚀、膨胀
# 7、找到图像中的最大轮廓，大概便是条形码
# 注：该方法做了关于图像梯度表示的假设，因此只对水平条形码有效。

def detect_bar(image):
    # 读入图片并灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算图像x方向和y方向的梯度
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # 利用x-gradient减去y-gradient，通过这一步减法操作，得到包含高水平梯度和低竖直梯度的图形区域
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 利用去噪仅关注条形码区域，使用9*9的内核对梯度图进行平均模糊，
    # 有助于平滑梯度表征的图形中的高频噪声，然后进行二值化
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # 对二值化图进行形态学操作，消除条形码竖杠之间的缝隙
    # 使用cv2.getStructuringElement构造一个长方形内核。这个内核的宽度大于长度，
    # 因此我们可以消除条形码中垂直条之间的缝隙。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 然后图像中还存在一些小斑点，于是用腐蚀和膨胀来消除旁边的小斑点
    #  腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    #  而膨胀操作将使剩余的白色像素扩张并重新增长回去。
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # 最后找图像中国条形码的轮廓
    (_, cnts, _)= cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # 通过对轮廓面积进行排序，找到面积最大的轮廓即为最外层轮廓
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # 计算最大轮廓的包围box
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # 将box画在原始图像中显示出来，这样便成功检测到了条形码
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    return image


if __name__ == '__main__':
    image = cv2.imread(r".\testpic\bar3.png")
    bar_image = detect_bar(image)
    cv2.imshow("bar", bar_image)
    cv2.imwrite("bar_image.jpg", bar_image)
    cv2.waitKey(0)