import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./images/test2.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray scale 변환

#plt.imshow(gray, cmap='gray') # 이미지 출력
plt.xticks([])
plt.yticks([])
#plt.show()

hist = cv2.calcHist([gray], [0], None, [256], [0,256]) # 원본 이미지 histogram 계산

plt.title('Before applying Histogram Equalization') # 원본 이미지 histogram graph 출력
plt.plot(hist, color='r', linewidth=1)
#plt.show()


equal = cv2.equalizeHist(gray) # Histogram Equalization 수행

#plt.imshow(equal, cmap='gray') # 변환된 이미지 출력
#plt.xticks([])
#plt.yticks([])
#plt.show()

hist = cv2.calcHist([equal], [0], None, [256], [0,256]) # equalized image의 histogram 계산
#result =cv2.cvtColor(equal, cv2.COLOR_GRAY2BGR) # Convert to RGB image
plt.title('After applying histogram equalization')  #equalized image의 histogram graph 출력
plt.plot(equal, color='r', linewidth=1)
#plt.show()

'''
# opencv 사용하여 이미지 출력
cv2.imshow('Equalized Image', equal)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

plt.imshow(cv2.cvtColor(equal, cv2.COLOR_GRAY2BGR))
#plt.show()
