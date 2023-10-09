import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import colorsys

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

# initializing the webcam
cap = cv2.VideoCapture(0)


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

while True:
    ret, image = cap.read()
    if not ret:
        break
    image_height, image_width, _ = image.shape

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(mp_image)

    # draw pose
    pose_landmarks_list = detection_result.pose_landmarks
    if not pose_landmarks_list:
        continue
    landmarks = pose_landmarks_list[0]

    '''
    ================================================================
    
    1. 양쪽 어깨 좌표 추출
    2. 추출한 좌표를 이용하여 radian 각도를 계산
        ➡️ 두 점 사이의 arctan 값을 계산, 어깨의 상대적인 위치에 따라 각도를 계산하기 위함
        🖥 radians = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])
    3. 계산된 radian 각도를 도(degree)로 변환
        🖥 angle = np.abs(radians * 180.0 / np.pi)
    
    ================================================================
    '''
    # Calculate angle between left shoulder and right shoulder
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    radians = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Draw current status on the screen
    status = "Good" if angle >= 170 else "Bad Posture"
    cv2.putText(image, 'Status: ' + status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw pose landmarks on the input image.
    annotated_image = np.copy(image)
    draw_landmark(annotated_image, landmarks, pairs)

    cv2.imshow('frame', annotated_image)
    # quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()