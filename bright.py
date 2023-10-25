import cv2
def adjust_brightness(image_path, brightness_factor, output_path):
    image = cv2.imread(image_path)
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Apply Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)
    cv2.imwrite(output_path, blurred_image)

    return output_path

def adjust_brightness_video(frame, brightness_factor):
    adjusted_image = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
    blurred_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)

    return blurred_image