import cv2
import numpy as np

def adjust_brightness(image_path, brightness_factor):
    image = cv2.imread(image_path)

    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Apply Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)

    cv2.imshow('Original Image', image)
    cv2.imshow('Adjusted Image', adjusted_image)
    cv2.imshow('Blurred Image', blurred_image)

    # cv2.imwrite('./images/adjusted_image_new.png', adjusted_image)
    cv2.imwrite('./images/blurred_image_new.png', blurred_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = './images/dark_test.jpg'
brightness_factor = 3.0  # Brightness factor

adjust_brightness(image_path, brightness_factor)