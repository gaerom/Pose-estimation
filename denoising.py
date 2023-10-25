import cv2

def denoising(image):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    cv2.imshow('original image', image)
    cv2.imshow('dst', dst)

    cv2.imwrite('./images/denoised.png', dst)

image = cv2.imread('./images/blurred_image_new.png')
denoising(image)

cv2.waitKey(0)
cv2.destroyAllWindows()