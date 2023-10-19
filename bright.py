import cv2
import numpy as np

def bright(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noimage = np.zeros(image.shape[:2], image.dtype)
    avg = cv2.mean(image)[0] / 2.0

    dst1 = cv2.scaleAdd(image, 10.0, noimage)
    dst2 = cv2.addWeighted(image, 2.0, noimage, 0, -avg)

    cv2.imshow('original', image)
    cv2.imshow('increase contrast', dst1) # 이게 더 명암 대비가 뚜렷


image_path = './images/test3.jpg'
bright(image_path)
cv2.waitKey(0)