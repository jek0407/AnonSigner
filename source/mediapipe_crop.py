import cv2
import mediapipe as mp
import numpy as np
import os
import imageio
import json

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def crop_and_resize_frame(frame, results, target_size=256):
    """수어 랜드마크를 기준으로 256x256 크롭 및 비율 유지"""
    h, w, _ = frame.shape
    all_landmarks = []

    # 랜드마크 위치 수집
    if results.pose_landmarks:
        all_landmarks += [(lm.x * w, lm.y * h) for lm in results.pose_landmarks.landmark]
    if results.face_landmarks:
        all_landmarks += [(lm.x * w, lm.y * h) for lm in results.face_landmarks.landmark]
    if results.left_hand_landmarks:
        all_landmarks += [(lm.x * w, lm.y * h) for lm in results.left_hand_landmarks.landmark]
    if results.right_hand_landmarks:
        all_landmarks += [(lm.x * w, lm.y * h) for lm in results.right_hand_landmarks.landmark]

    # 랜드마크가 없는 경우 원본 프레임 반환
    if not all_landmarks:
        return cv2.resize(frame, (target_size, target_size))

    # 랜드마크의 경계 계산
    x_min = int(min([p[0] for p in all_landmarks]))
    y_min = int(min([p[1] for p in all_landmarks]))
    x_max = int(max([p[0] for p in all_landmarks]))
    y_max = int(max([p[1] for p in all_landmarks]))

    # 확장된 ROI 설정 (여유 공간 추가)
    padding = 50
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # ROI 크롭
    cropped_frame = frame[y_min:y_max, x_min:x_max]

    # 비율 유지하며 리사이즈
    cropped_h, cropped_w, _ = cropped_frame.shape
    scale = min(target_size / cropped_w, target_size / cropped_h)
    new_w, new_h = int(cropped_w * scale), int(cropped_h * scale)
    resized_frame = cv2.resize(cropped_frame, (new_w, new_h))

    # 패딩 추가하여 256x256 정사각형 생성
    padded_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded_frame[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_frame

    return padded_frame

# 입력 및 출력 설정
idx = 1
output_dir = f"./output/{idx}"
os.makedirs(output_dir, exist_ok=True)

input_video_path = f"./input/sign_language.mp4"
output_video_path = f"{output_dir}/sign_language_cropped.mp4"

cap = cv2.VideoCapture(input_video_path)

# 동영상 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (256, 256))

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # RGB 변환 및 MediaPipe 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # 프레임 크롭 및 리사이즈
        cropped_frame = crop_and_resize_frame(frame, results, target_size=256)

        # 동영상 저장
        out.write(cropped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Cropped and resized video saved at: {output_video_path}")
