import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

# argparse를 사용하여 입력 및 출력 경로 설정
def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe Pose Estimation on Video")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input/sign_language.mp4", 
        help="Path to the input video file (default: data/input/sign_language.mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output/01_mediapipe_pose", 
        help="Directory to save the output video file (default: data/output/01_mediapipe_pose)"
    )
    return parser.parse_args()

args = parse_args()

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 입력 및 출력 경로 설정
input_video_path = args.input
output_dir = args.output  

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 출력 파일명 설정
output_video_path = os.path.join(output_dir, "pose_landmark.mp4")

# 동영상 처리
cap = cv2.VideoCapture(input_video_path)

# 동영상 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

with mp_pose.Pose(
    static_image_mode=False,  # 비디오 처리 모드
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Pose 처리
        results = pose.process(frame_rgb)

        # 랜드마크 그리기
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            # 동영상 파일로 저장
            out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Pose landmarks video saved at: {output_video_path}")
