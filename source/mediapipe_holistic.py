import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

# argparse를 사용하여 입력 및 출력 경로 설정
def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe Holistic Estimation on Video")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input/sign_language.mp4", 
        help="Path to the input video file (default: data/input/sign_language.mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output/02_mediapipe_holistic", 
        help="Directory to save the output video file (default: data/output/02_mediapipe_holistic)"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="XVID",  # MP4를 위한 기본 코덱 설정
        choices=["XVID", "mp4v", "MJPG"],
        help="Codec for the output video (default: XVID)"
    )
    return parser.parse_args()

args = parse_args()

# MediaPipe Holistic 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 입력 및 출력 경로 설정
input_video_path = args.input
output_dir = args.output  
os.makedirs(output_dir, exist_ok=True)

output_video_path = os.path.join(output_dir, "sign_language_holistic.mp4")

# 동영상 처리
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError(f"❌ Unable to open video file: {input_video_path}")

# 원본 해상도 유지
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 동영상 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*args.codec)  # 사용자 지정 코덱
out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

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

        # BGR 이미지를 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Holistic 처리
        results = holistic.process(frame_rgb)

        # 랜드마크 그리기
        annotated_frame = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

        if results.segmentation_mask is not None:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros_like(frame, dtype=np.uint8)
            bg_image[:] = (0, 0, 0)  # 배경을 검정색으로 설정
            annotated_frame = np.where(condition, annotated_frame, bg_image)

        # 원본 해상도로 동영상 저장
        out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Holistic landmarks video saved at: {output_video_path}")