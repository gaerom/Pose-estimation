import cv2
import os
### Convert to BGR
def cvtImageToBGR(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_path) and (input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))):
            image = cv2.imread(input_path)
            imgBGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            output_path = os.path.join(output_dir, f'bgr_{filename}')
            cv2.imwrite(output_path, imgBGR)