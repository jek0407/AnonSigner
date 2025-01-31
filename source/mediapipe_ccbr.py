import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

# MediaPipe Holistic 초기화
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# argparse를 사용해 입력 및 출력 경로를 동적으로 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with MediaPipe Holistic")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input/sign_language.mp4", 
        help="Path to the input video file (default: data/input/sign_language.mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output/03_mediapipe_ccbr", 
        help="Directory to save output video and images (default: data/output/03_mediapipe_ccbr)"
    )
    parser.add_argument("--debug", action="store_true", help="Save individual frames as images for debugging")
    return parser.parse_args()

args = parse_args()

# 입력 및 출력 설정
input_video_path = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, "sign_language_ccbr.mp4")

# 동영상 처리
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError(f"Unable to open video file: {input_video_path}")

# 원본 비율 확인
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 원본 비율을 유지한 상태로 크기 조정 (긴 변을 256으로 설정)
if orig_width > orig_height:
    width = 256
    height = int((orig_height / orig_width) * 256)
else:
    height = 256
    width = int((orig_width / orig_height) * 256)

# 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 랜드마크 그룹별 색상 매핑 함수
def get_color(group, index, max_index):
    blue_values = {
        "left_hand": 0.0,
        "right_hand": 0.99,
        "pose": 0.33,
        "face": 0.66
    }
    fixed_blue = blue_values.get(group, 0.5)

    red = int((index / max_index) * 255)
    green = int(((max_index - index) / max_index) * 255)

    return (red, green, int(fixed_blue * 255))

# 랜드마크를 그룹별로 캔버스에 그림
def draw_landmarks_with_group_colors(landmarks, canvas, width, height, group_name):
    max_index = len(landmarks.landmark) - 1
    for idx, lm in enumerate(landmarks.landmark):
        x, y = int(lm.x * width), int(lm.y * height)
        color = get_color(group_name, idx, max_index)
        cv2.circle(canvas, (x, y), 1, color, -1)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 조정 (비율 유지)
        frame = cv2.resize(frame, (width, height))

        # BGR 이미지를 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Holistic 처리
        results = holistic.process(frame_rgb)

        # 새 캔버스 생성 (검은 배경)
        landmark_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # 랜드마크 그룹별 색상 적용
        if results.pose_landmarks:
            draw_landmarks_with_group_colors(results.pose_landmarks, landmark_canvas, width, height, "pose")

        if results.face_landmarks:
            draw_landmarks_with_group_colors(results.face_landmarks, landmark_canvas, width, height, "face")

        if results.left_hand_landmarks:
            draw_landmarks_with_group_colors(results.left_hand_landmarks, landmark_canvas, width, height, "left_hand")

        if results.right_hand_landmarks:
            draw_landmarks_with_group_colors(results.right_hand_landmarks, landmark_canvas, width, height, "right_hand")

        # 디버그 모드에서만 개별 프레임 저장
        if args.debug:
            landmark_image_path = os.path.join(output_dir, f"landmark_combined_{frame_index:04d}.png")
            cv2.imwrite(landmark_image_path, landmark_canvas)

        # 동영상 파일로 저장
        out.write(landmark_canvas)
        frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"ccbr video saved at: {output_video_path}")
if args.debug:
    print(f"Landmark images saved in directory: {output_dir}")
