import cv2
import mediapipe as mp
import numpy as np
import json
import os
import imageio

# MediaPipe Holistic 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 입력 및 출력 경로 설정
idx = 1
output_dir = f"./output/{idx}"  # 출력 디렉토리
os.makedirs(output_dir, exist_ok=True)

input_video_path = f"./input/sign_language_{idx}.mp4"  # 입력 동영상 파일
output_video_path = f"{output_dir}/sign_language_holistic.mp4"  # 결과 저장 파일
pose_landmark_path = f"{output_dir}/pose_landmarks.json"  # Pose 랜드마크 저장 파일
face_landmark_path = f"{output_dir}/face_landmarks.json"  # Face 랜드마크 저장 파일
hand_landmark_path = f"{output_dir}/hand_landmarks.json"  # Hand 랜드마크 저장 파일
segmentation_mask_path = f"{output_dir}/segmentation_mask.json"  # Segmentation Mask 저장 파일
pose_image_path = f"{output_dir}/pose_landmarks.png"  # Pose 랜드마크 이미지 저장 파일
face_image_path = f"{output_dir}/face_landmarks.png"  # Face 랜드마크 이미지 저장 파일
hand_image_path = f"{output_dir}/hand_landmarks.png"  # Hand 랜드마크 이미지 저장 파일
segmentation_image_path = f"{output_dir}/segmentation_mask.png"  # Segmentation Mask 이미지 저장 파일

# 동영상 처리
cap = cv2.VideoCapture(input_video_path)

# 동영상 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = 640  # 해상도를 640x480으로 고정
frame_height = 480
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_index = 50

with mp_holistic.Holistic(
    static_image_mode=False,  # 비디오 처리 모드
    model_complexity=2,
    enable_segmentation=True,  # 분할 마스크 활성화
    refine_face_landmarks=True,  # 얼굴 랜드마크 정교화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    # frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # # 해상도 확인
        # height, width, _ = frame.shape
        # frame_count += 1
        # print(f"Frame {frame_count}: Width = {width}, Height = {height}")

        # 해상도 축소
        frame = cv2.resize(frame, (640, 480))

        # BGR 이미지를 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Holistic 처리
        results = holistic.process(frame_rgb)

        # 랜드마크 저장 및 이미지 생성 (첫 번째 프레임만)
        if frame_index == 0:
            pose_landmarks = []
            face_landmarks = []
            hand_landmarks = {
                "left_hand_landmarks": [],
                "right_hand_landmarks": []
            }
            segmentation_mask = None

            if results.pose_landmarks:
                pose_landmarks = [
                    {
                        "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility
                    } for lm in results.pose_landmarks.landmark
                ]
                # Pose 랜드마크 이미지 저장
                annotated_pose = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_pose,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                imageio.imwrite(pose_image_path, cv2.cvtColor(annotated_pose, cv2.COLOR_BGR2RGB))

            if results.face_landmarks:
                face_landmarks = [
                    {
                        "x": lm.x, "y": lm.y, "z": lm.z
                    } for lm in results.face_landmarks.landmark
                ]
                # Face 랜드마크 이미지 저장
                annotated_face = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_face,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                imageio.imwrite(face_image_path, cv2.cvtColor(annotated_face, cv2.COLOR_BGR2RGB))

            if results.left_hand_landmarks or results.right_hand_landmarks:
                hand_annotated_frame = frame.copy()

                if results.left_hand_landmarks:
                    hand_landmarks["left_hand_landmarks"] = [
                        {
                            "x": lm.x, "y": lm.y, "z": lm.z
                        } for lm in results.left_hand_landmarks.landmark
                    ]
                    mp_drawing.draw_landmarks(
                        hand_annotated_frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                    )

                if results.right_hand_landmarks:
                    hand_landmarks["right_hand_landmarks"] = [
                        {
                            "x": lm.x, "y": lm.y, "z": lm.z
                        } for lm in results.right_hand_landmarks.landmark
                    ]
                    mp_drawing.draw_landmarks(
                        hand_annotated_frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                    )
                # Hand 랜드마크 이미지 저장
                imageio.imwrite(hand_image_path, cv2.cvtColor(hand_annotated_frame, cv2.COLOR_BGR2RGB))

            if results.segmentation_mask is not None:
                segmentation_mask = results.segmentation_mask.tolist()
                # Segmentation Mask 이미지 저장
                segmentation_image = (np.array(results.segmentation_mask) * 255).astype(np.uint8)
                imageio.imwrite(segmentation_image_path, segmentation_image)

            # JSON 파일로 저장
            with open(pose_landmark_path, "w") as f:
                json.dump({"pose_landmarks": pose_landmarks}, f, indent=4)

            with open(face_landmark_path, "w") as f:
                json.dump({"face_landmarks": face_landmarks}, f, indent=4)

            with open(hand_landmark_path, "w") as f:
                json.dump(hand_landmarks, f, indent=4)

            with open(segmentation_mask_path, "w") as f:
                json.dump({"segmentation_mask": segmentation_mask}, f, indent=4)

            # 랜드마크 수 출력
            pose_count = len(pose_landmarks)
            face_count = len(face_landmarks)
            left_hand_count = len(hand_landmarks["left_hand_landmarks"])
            right_hand_count = len(hand_landmarks["right_hand_landmarks"])
            total_count = pose_count + face_count + left_hand_count + right_hand_count

            print(f"Pose landmarks: {pose_count}")
            print(f"Face landmarks: {face_count}")
            print(f"Left hand landmarks: {left_hand_count}")
            print(f"Right hand landmarks: {right_hand_count}")
            print(f"Total landmarks: {total_count}")

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

        # 동영상 파일로 저장
        out.write(annotated_frame)
        frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Holistic landmarks video saved at: {output_video_path}")
print(f"Pose landmarks saved at: {pose_landmark_path}")
print(f"Face landmarks saved at: {face_landmark_path}")
print(f"Hand landmarks saved at: {hand_landmark_path}")
print(f"Segmentation mask saved at: {segmentation_mask_path}")
print(f"Pose landmark image saved at: {pose_image_path}")
print(f"Face landmark image saved at: {face_image_path}")
print(f"Hand landmark image saved at: {hand_image_path}")
print(f"Segmentation mask image saved at: {segmentation_image_path}")
