import cv2

image = cv2.imread('./images/result1.png')
dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

cv2.imshow('original image', image)
cv2.imshow('dst', dst)

cv2.imwrite('./images/result2.png', dst)