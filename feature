import cv2 as cv
import numpy as np

img = cv.imread('image4.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Harris角点检测
gray = np.float32(gray)
dst = cv.cornerHarris(gray,blockSize=2,ksize=3,k=0.04)

# 标记角点
dst = cv.dilate(dst,None)
img[dst>0.001*dst.max()] = [0,255,0]

cv.imshow('Harris Corners',img)
cv.waitKey(0)
cv.destroyAllWindows()
