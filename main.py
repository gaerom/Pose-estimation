import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import colorsys
import os
from lpFilter import create_low_pass_filter, apply_low_pass_filter

# Import adjust_brightness and denoising functions
from bright import adjust_brightness_video
from denoising import video_denoising

# init mediapipe task
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a low-pass filter
fps = 25  # Frame rate
cutoff_frequency = 0.5  # Desired cutoff frequency
b, a = create_low_pass_filter(fps, cutoff_frequency)

# Initialize MediaPipe
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
        image = cv2.circle(image, (landmark_x, landmark_y), 20, (255, 255, 255), -1)

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

        image = cv2.line(image, (landmark_x_1, landmark_y_1), (landmark_x_2, landmark_y_2), colors[pair_id], 10)

# Open the video file
video_path = './images/test_video.mov'  # Specify the path to your video file
cap = cv2.VideoCapture(video_path)

# Create an output video writer
output_path = './result/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 밝게 만들기
    brightness_factor = 3.0
    brightened_frame = adjust_brightness_video(frame, brightness_factor)

    # 이미지 노이즈 제거
    denoised_frame = video_denoising(brightened_frame)

    image_height, image_width, _ = denoised_frame.shape

    # Convert the frame to MediaPipe’s Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=denoised_frame)
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

    status = "Good" if angle >= 175 else "Bad Posture"
    font_scale = 0.5
    font_color = (0, 0, 255)
    font_thickness = 0.5
    cv2.putText(denoised_frame, 'Status: ' + status, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2, cv2.LINE_AA)

    annotated_frame = np.copy(denoised_frame)
    draw_landmark(annotated_frame, landmarks, pairs)

    # Apply low-pass filter to keypoints
    for i in range(len(landmarks)):
        x_values = [landmark.x for landmark in landmarks]
        y_values = [landmark.y for landmark in landmarks]

        filtered_x = apply_low_pass_filter(x_values, b, a)
        filtered_y = apply_low_pass_filter(y_values, b, a)

        for j in range(len(landmarks)):
            landmarks[j].x = filtered_x[j]
            landmarks[j].y = filtered_y[j]

    out.write(annotated_frame)
    cv2.imshow('Video', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()