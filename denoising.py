import cv2

def denoising(image_path, output_path):
    image = cv2.imread(image_path)

    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    cv2.imwrite(output_path, dst)

    return output_path

def video_denoising(frame):
    dst = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    return dst