import cv2
import mediapipe as mp
import numpy as np
import os
import json
import imageio
import argparse

# argparse를 사용하여 입력 및 출력 경로를 동적으로 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Extract and save MediaPipe Holistic landmarks and images")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input/sign_language.mp4", 
        help="Path to the input video file (default: data/input/sign_language.mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output/05_reference", 
        help="Directory to save output JSON and PNG files (default: data/output/05_reference)"
    )
    return parser.parse_args()

args = parse_args()

# MediaPipe Holistic 초기화
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 입력 및 출력 설정
input_video_path = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

pose_landmark_path = os.path.join(output_dir, "pose_landmarks.json")
face_landmark_path = os.path.join(output_dir, "face_landmarks.json")
hand_landmark_path = os.path.join(output_dir, "hand_landmarks.json")
segmentation_mask_path = os.path.join(output_dir, "segmentation_mask.json")
pose_image_path = os.path.join(output_dir, "pose_landmarks.png")
face_image_path = os.path.join(output_dir, "face_landmarks.png")
hand_image_path = os.path.join(output_dir, "hand_landmarks.png")
segmentation_image_path = os.path.join(output_dir, "segmentation_mask.png")

# 동영상 처리
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError(f"❌ Unable to open video file: {input_video_path}")

frame_index = 0

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

        # 원본 해상도를 유지하여 처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame_rgb)

        # 50번째 프레임만 처리
        if frame_index == 50:
            pose_landmarks = []
            face_landmarks = []
            hand_landmarks = {"left_hand_landmarks": [], "right_hand_landmarks": []}
            segmentation_mask = None

            if results.pose_landmarks:
                pose_landmarks = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in results.pose_landmarks.landmark
                ]
                annotated_pose = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_pose,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
                cv2.imwrite(pose_image_path, annotated_pose)

            if results.face_landmarks:
                face_landmarks = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in results.face_landmarks.landmark
                ]
                annotated_face = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_face,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )
                cv2.imwrite(face_image_path, annotated_face)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                hand_annotated_frame = frame.copy()

                if results.left_hand_landmarks:
                    hand_landmarks["left_hand_landmarks"] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in results.left_hand_landmarks.landmark
                    ]
                    mp_drawing.draw_landmarks(
                        hand_annotated_frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )

                if results.right_hand_landmarks:
                    hand_landmarks["right_hand_landmarks"] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in results.right_hand_landmarks.landmark
                    ]
                    mp_drawing.draw_landmarks(
                        hand_annotated_frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
                    )

                cv2.imwrite(hand_image_path, hand_annotated_frame)

            if results.segmentation_mask is not None:
                segmentation_mask = results.segmentation_mask.tolist()
                segmentation_image = (np.array(results.segmentation_mask) * 255).astype(np.uint8)
                imageio.imwrite(segmentation_image_path, segmentation_image)

            # JSON 파일 저장
            with open(pose_landmark_path, "w") as f:
                json.dump({"pose_landmarks": pose_landmarks}, f, indent=4)

            with open(face_landmark_path, "w") as f:
                json.dump({"face_landmarks": face_landmarks}, f, indent=4)

            with open(hand_landmark_path, "w") as f:
                json.dump(hand_landmarks, f, indent=4)

            with open(segmentation_mask_path, "w") as f:
                json.dump({"segmentation_mask": segmentation_mask}, f, indent=4)

            # 랜드마크 개수 출력
            pose_count = len(pose_landmarks)
            face_count = len(face_landmarks)
            left_hand_count = len(hand_landmarks["left_hand_landmarks"])
            right_hand_count = len(hand_landmarks["right_hand_landmarks"])
            total_count = pose_count + face_count + left_hand_count + right_hand_count

            print(f"✅ Pose landmarks: {pose_count}")
            print(f"✅ Face landmarks: {face_count}")
            print(f"✅ Left hand landmarks: {left_hand_count}")
            print(f"✅ Right hand landmarks: {right_hand_count}")
            print(f"✅ Total landmarks: {total_count}")

            break  # 루프 종료

        frame_index += 1

cap.release()
cv2.destroyAllWindows()

print(f"Pose landmarks saved at: {pose_landmark_path}")
print(f"Face landmarks saved at: {face_landmark_path}")
print(f"Hand landmarks saved at: {hand_landmark_path}")
print(f"Segmentation mask saved at: {segmentation_mask_path}")
print(f"Pose landmark image saved at: {pose_image_path}")
print(f"Face landmark image saved at: {face_image_path}")
print(f"Hand landmark image saved at: {hand_image_path}")
print(f"Segmentation mask image saved at: {segmentation_image_path}")
