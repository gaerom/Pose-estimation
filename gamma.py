import cv2
import numpy as np

image = cv2.imread('./images/test3.jpg')
image = cv2.resize(image, dsize=(0,0), fx=0.25, fy=0.25)

def gamma(f, gamma=1.0):
    f1 = f / 255.0
    return np.uint8(255*(f1**gamma))

gc = np.hstack((gamma(image, 0.5), gamma(image, 0.75), gamma(image, 1.0), gamma(image, 2.0), gamma(image, 3.0)))
cv2.imwrite('./gamma/gamma_result.png', gc)
cv2.imshow('gamma', gc)


cv2.waitKey()
cv2.destroyAllWindows()