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
        default="data/output/04_mediapipe_ccbr_face", 
        help="Directory to save output videos (default: data/output/04_mediapipe_ccbr_face)"
    )
    return parser.parse_args()

args = parse_args()

# 입력 및 출력 설정
input_video_path = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)
output_face_video_path = os.path.join(output_dir, "sign_language_ccbr_face.mp4")

# 동영상 처리
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError(f"Unable to open video file: {input_video_path}")

# 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = 256
frame_height = 256
face_out = cv2.VideoWriter(output_face_video_path, fourcc, fps, (frame_width, frame_height))

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
def draw_landmarks_with_group_colors(landmarks, canvas, group_name):
    max_index = len(landmarks.landmark) - 1
    for idx, lm in enumerate(landmarks.landmark):
        x = int(lm.x * 256)
        y = int(lm.y * 256)
        color = get_color(group_name, idx, max_index)
        cv2.circle(canvas, (x, y), 1, color, -1)

with mp_holistic.Holistic(
    static_image_mode=False,  # 비디오 처리 모드
    model_complexity=2,
    enable_segmentation=True,  # 분할 마스크 활성화
    refine_face_landmarks=True,  # 얼굴 랜드마크 정교화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Holistic 처리
        results = holistic.process(frame_rgb)

        # 새 캔버스 생성 (검은 배경)
        face_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # 얼굴 랜드마크 확대 및 적용
        if results.face_landmarks:
            xs = [lm.x * frame.shape[1] for lm in results.face_landmarks.landmark]
            ys = [lm.y * frame.shape[0] for lm in results.face_landmarks.landmark]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            box_size = max(x_max - x_min, y_max - y_min) * 1.2

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            start_x = max(0, center_x - box_size / 2)
            start_y = max(0, center_y - box_size / 2)
            end_x = min(frame.shape[1], center_x + box_size / 2)
            end_y = min(frame.shape[0], center_y + box_size / 2)

            for idx, lm in enumerate(results.face_landmarks.landmark):
                x = int((lm.x * frame.shape[1] - start_x) / (end_x - start_x) * frame_width)
                y = int((lm.y * frame.shape[0] - start_y) / (end_y - start_y) * frame_height)
                color = get_color("face", idx, len(results.face_landmarks.landmark) - 1)
                cv2.circle(face_canvas, (x, y), 1, color, -1)

        # 동영상 파일로 저장
        face_out.write(face_canvas)

cap.release()
face_out.release()
cv2.destroyAllWindows()

print(f"Face landmarks video saved at: {output_face_video_path}")
