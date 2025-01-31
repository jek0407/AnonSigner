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
        default="./input/sign_language.mp4", 
        help="Path to the input video file (default: ./input/sign_language.mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./output/1/ccbr", 
        help="Directory to save output video and images (default: ./output/1/ccbr)"
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

# 동영상 해상도 동적으로 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 랜드마크 그룹별 색상 매핑 함수
def get_color(group, index, max_index):
    """
    그룹과 인덱스에 따라 색상을 반환.
    - group: 랜드마크 그룹 (left_hand, right_hand, pose, face)
    - index: 현재 랜드마크의 포인트 번호
    - max_index: 그룹 내 최대 랜드마크 번호
    """
    blue_values = {
        "left_hand": 0.0,
        "right_hand": 0.99,
        "pose": 0.33,
        "face": 0.66
    }
    fixed_blue = blue_values.get(group, 0.5)  # 기본값 0.5 (예외 처리)

    # RED와 GREEN 값을 index에 따라 선형 분포
    red = int((index / max_index) * 255)
    green = int(((max_index - index) / max_index) * 255)

    return (red, green, int(fixed_blue * 255))

# 랜드마크를 그룹별로 캔버스에 그림
def draw_landmarks_with_group_colors(landmarks, canvas, width, height, group_name):
    max_index = len(landmarks.landmark) - 1
    for idx, lm in enumerate(landmarks.landmark):
        x, y = int(lm.x * width), int(lm.y * height)
        color = get_color(group_name, idx, max_index)
        cv2.circle(canvas, (x, y), 3, color, -1)

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
