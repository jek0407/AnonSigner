import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import time
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Process all videos in a directory with MediaPipe Holistic")
    parser.add_argument("--input_dir", type=str, default="/home/mmlab/MMLAB/sl-vae/data/original",
                        help="Path to the directory containing input video files")
    parser.add_argument("--output_dir", type=str, default="../data/color-coded_ver2",
                        help="Directory to save output videos")
    parser.add_argument("--debug", action="store_true", help="Save individual frames as images for debugging")
    return parser.parse_args()

args = parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def get_color(group, index, max_index):
    blue_values = {"left_hand": 0.0, "right_hand": 0.99, "pose": 0.33, "face": 0.66}
    fixed_blue = blue_values.get(group, 0.5)
    red = int((index / max_index) * 255)
    green = int(((max_index - index) / max_index) * 255)
    return (red, green, int(fixed_blue * 255))

def draw_landmarks_with_group_colors(landmarks, canvas, width, height, group_name, connections=None):
    max_index = len(landmarks.landmark) - 1
    points = []

    # 각 랜드마크 점을 캔버스에 그림
    for idx, lm in enumerate(landmarks.landmark):
        x, y = int(lm.x * width), int(lm.y * height)
        color = get_color(group_name, idx, max_index)
        cv2.circle(canvas, (x, y), 3, color, -1)
        points.append((x, y))

    # 점을 선으로 연결
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(canvas, points[start_idx], points[end_idx], (255, 255, 255), 1)  # 흰색 선

def process_video(input_video_path, output_video_path, debug=False):
    print(f"Starting processing: {input_video_path}")
    start_time = time.time()
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {input_video_path}")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width, height = orig_width, orig_height  # 원본 해상도 유지
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        ) as holistic:
        
        frame_index = 0
        # frame_gpu = cv2.cuda_GpuMat()  # OpenCV CUDA 가속

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
            
            if results.pose_landmarks:
                draw_landmarks_with_group_colors(results.pose_landmarks, landmark_canvas, width, height, "pose", mp_holistic.POSE_CONNECTIONS)
            if results.face_landmarks:
                draw_landmarks_with_group_colors(results.face_landmarks, landmark_canvas, width, height, "face", mp_holistic.FACEMESH_TESSELATION)
            if results.left_hand_landmarks:
                draw_landmarks_with_group_colors(results.left_hand_landmarks, landmark_canvas, width, height, "left_hand", mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                draw_landmarks_with_group_colors(results.right_hand_landmarks, landmark_canvas, width, height, "right_hand", mp_holistic.HAND_CONNECTIONS)

            if debug:
                frame_path = os.path.join(output_video_path.replace(".mp4", ""), f"landmark_{frame_index:04d}.png")
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                cv2.imwrite(frame_path, landmark_canvas)
            
            out.write(landmark_canvas)
            frame_index += 1
            if frame_index % 50 == 0:
                print(f"Processing frame {frame_index} in {input_video_path}")

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"Processed video saved at: {output_video_path}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    if debug:
        print(f"Debug frames saved in directory: {output_video_path.replace('.mp4', '')}")

def process_videos_parallel(video_files):
    """멀티프로세싱으로 여러 개의 비디오를 동시에 처리"""
    num_workers = min(multiprocessing.cpu_count(), len(video_files))  # CPU 개수만큼 워커 할당
    print(f"Using {num_workers} parallel workers for video processing.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_video_wrapper, video_files)

def process_video_wrapper(video_file):
    """process_video 함수의 병렬 실행을 위한 래퍼 함수"""
    input_video_path, output_video_path = video_file
    process_video(input_video_path, output_video_path, debug=args.debug)

if __name__ == "__main__":
    # ✅ 기존에 변환된 파일 목록 확인 (이미 처리된 파일은 제외)
    processed_files = {f.replace("_ccbr.mp4", ".mp4") for f in os.listdir(output_dir) if f.endswith("_ccbr.mp4")}
    all_files = {f for f in os.listdir(input_dir) if f.endswith(".mp4")}
    pending_files = list(all_files - processed_files)  # 처리되지 않은 파일만 실행

    print(f"✅ 이미 처리된 파일 수: {len(processed_files)}")
    print(f"⏳ 아직 처리되지 않은 파일 수: {len(pending_files)}")

    # 처리되지 않은 파일만 실행
    video_files = [(os.path.join(input_dir, f), os.path.join(output_dir, f.replace(".mp4", "_ccbr.mp4"))) 
                   for f in pending_files]

    if video_files:
        print(f"🚀 {len(video_files)}개의 파일을 다시 처리합니다.")
        process_videos_parallel(video_files)
    else:
        print("✅ 모든 파일이 이미 처리되어 추가 실행할 필요가 없습니다.")