import cv2

image = cv2.imread('./images/test2.jpeg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert to HSV
h, s, v = cv2.split(hsv) # 각각의 채널로 컬러 영상을 분리
equalV = cv2.equalizeHist(v) # value(명도)를 equalization

new_hsv = cv2.merge([h, s, equalV]) # 위 equalized v를 사용
hsv2bgr = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR) # Convert to BGR



yCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y, Cr, Cb = cv2.split(yCrCb)
equalY = cv2.equalizeHist(y) # y(휘도)를 equalization

new_yCrCb = cv2.merge([equalY, Cr, Cb])
yCrCb2bgr = cv2.cvtColor(new_yCrCb, cv2.COLOR_YCrCb2BGR)

cv2.imshow('original', image)
cv2.imshow('HSV', hsv2bgr)
cv2.imshow('yCrCb', yCrCb2bgr)
cv2.waitKey()
cv2.destroyAllWindows()

