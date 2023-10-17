import cv2
import matplotlib.pyplot as plt

image = cv2.imread('sports.jpeg')
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB로 변환

### 원본 이미지 histogram 계산
hist = cv2.calcHist([imageRGB], [0], None, [256], [0,256])
plt.title('Original Image') # 원본 이미지 histogram graph 출력
plt.plot(hist, color='r', linewidth=1), plt.show()
cv2.imwrite('./sports equalized image/original.png', imageRGB) # 이미지 저장


### Convert to HSV
hsv = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv) # 각각의 채널로 컬러 영상을 분리
equalV = cv2.equalizeHist(v) # value(명도)를 equalization

new_hsv = cv2.merge([h, s, equalV]) # 위 equalized v를 사용
hsv2rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB) # Convert to BGR

hist_hsv = cv2.calcHist([hsv2rgb], [0], None, [256], [0,256]) # HSV histogram 계산
plt.title('HSV')
plt.plot(hist_hsv, color='g', linewidth=1), plt.show()
cv2.imwrite('./sports equalized image/hsv.png', hsv2rgb)



### Convert to yCrCb
yCrCb = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2YCrCb)
y, Cr, Cb = cv2.split(yCrCb)
equalY = cv2.equalizeHist(y) # y(휘도)를 equalization

new_yCrCb = cv2.merge([equalY, Cr, Cb])
yCrCb2rgb = cv2.cvtColor(new_yCrCb, cv2.COLOR_YCrCb2RGB)

hist_ycrcb = cv2.calcHist([yCrCb2rgb], [0], None, [256], [0,256]) # YCrCb histogram 계산
plt.title('Result')
plt.plot(hist_ycrcb, color='b', linewidth=1), plt.show()
cv2.imwrite('./sports equalized image/ycrcb.png', yCrCb2rgb)



### 결과 출력
cv2.imshow('original', image)
cv2.imshow('HSV', hsv2rgb)
cv2.imshow('yCrCb', yCrCb2rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
