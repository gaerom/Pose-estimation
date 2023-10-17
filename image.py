import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import colorsys
import os

# init mediapipe task
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_pose = mp.solutions.pose

model_path = '/Users/saeromkim/pose/pose_landmarker_full.task'
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, num_poses=1)

detector = vision.PoseLandmarker.create_from_options(options)

# for drawing limbs
pairs = list(mp.solutions.pose.POSE_CONNECTIONS)
colors = [tuple(int(255 * i) for i in colorsys.hsv_to_rgb(x / len(pairs), 1.0, 1.0)) for x in range(len(pairs))]


def draw_landmark(image, landmarks, pairs):
    # draw circles
    for idx, landmark in enumerate(landmarks):
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        visibility = landmark.visibility
        image = cv2.circle(image, (landmark_x, landmark_y), 5, (255, 255, 255), -1)

    # draw limbs
    for pair_id, pair in enumerate(pairs):
        idx1 = pair[0]
        idx2 = pair[1]

        landmark_x_1 = int(landmarks[idx1].x * image_width)
        landmark_y_1 = int(landmarks[idx1].y * image_height)
        visibility_1 = landmarks[idx1].visibility

        landmark_x_2 = int(landmarks[idx2].x * image_width)
        landmark_y_2 = int(landmarks[idx2].y * image_height)
        visibility_2 = landmarks[idx1].visibility

        image = cv2.line(image, (landmark_x_1, landmark_y_1), (landmark_x_2, landmark_y_2), colors[pair_id], 2)



image_path = './sports equalized image/original.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=4, fy=4)

image_height, image_width, _ = image.shape

# Convert the image to MediaPipe’s Image format
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
detection_result = detector.detect(mp_image)


pose_landmarks_list = detection_result.pose_landmarks

if detection_result.pose_landmarks:
    print('yes')
else:
    print('no detection')

landmarks = pose_landmarks_list[0]

# 어깨 각도 계산
left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

radians = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])
angle = np.abs(radians * 180.0 / np.pi)


status = "Good" if angle >= 175 else "Bad Posture"     # 170
font_scale = 0.5  # You can adjust this scale as needed
font_color = (0, 0, 255)
font_thickness = 0.5
cv2.putText(image, 'Status: ' + status, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, font_thickness, font_color, 2, cv2.LINE_AA)


annotated_image = np.copy(image)
draw_landmark(annotated_image, landmarks, pairs)

cv2.imshow('Image', annotated_image)



### Save the result
result_dir = './results'
base_name = os.path.basename(image_path)
name, ext = os.path.splitext(base_name)
result_fname = f'result_{name}{ext}'

result_path = os.path.join(result_dir, result_fname)


cv2.imwrite(result_path, annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()