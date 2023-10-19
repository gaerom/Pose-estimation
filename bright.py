import cv2
import numpy as np

def adjust_brightness(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 사람 영역을 위한 mask 생성
    mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)[1]

    # mask 반전
    mask = cv2.bitwise_not(mask)

    # 사람 영역에 밝기 조절을 적용
    avg = cv2.mean(image)[0] / 2.0
    person_brightness = cv2.addWeighted(gray_image, 2.0, np.zeros_like(gray_image), 0, -avg)

    # 조정된 사람 영역과 배경을 결합
    result_image = cv2.add(cv2.bitwise_and(image, image, mask=mask), cv2.bitwise_and(cv2.cvtColor(person_brightness, cv2.COLOR_GRAY2BGR), cv2.cvtColor(person_brightness, cv2.COLOR_GRAY2BGR), mask=mask))

    cv2.imshow('original', image)
    cv2.imshow('adjusted image', result_image)

image_path = './images/test3.jpg'
adjust_brightness(image_path)
cv2.waitKey(0)
