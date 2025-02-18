#!/usr/bin/env python3
import cv2
import os
import argparse
import multiprocessing
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--input_dir", type=str, default="/media/mmlab/mldisk2/sign_language/out_3",
                        help="Path to the directory containing input video files")
    parser.add_argument("--output_dir", type=str, default="/home/mmlab/Data/color-coded-frames",
                        help="Directory to save extracted frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="Number of frames per second to extract (default: 30)")
    return parser.parse_args()

args = parse_args()
setup_logging()

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

def extract_frames(video_file):
    """
    비디오에서 프레임을 시간 기반으로 추출하여 저장합니다.
    """
    input_video_path, output_folder = video_file
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"🔄 Extracting frames from {input_video_path} at {args.fps} FPS.")

    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logging.error(f"❌ Cannot open video: {input_video_path}")
            return

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            logging.warning(f"Invalid FPS for {input_video_path}. Defaulting to 30.")
            orig_fps = 30

        desired_fps = args.fps
        frame_interval_sec = 1 / desired_fps

        frame_count = 0
        saved_count = 0
        next_capture_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / orig_fps
            if current_time >= next_capture_time:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                next_capture_time += frame_interval_sec

            frame_count += 1
    except Exception as e:
        logging.error(f"Error extracting frames from {input_video_path}: {e}")
    finally:
        cap.release()
    logging.info(f"✅ Extracted {saved_count} frames from {input_video_path}")

def process_videos_parallel(video_files):
    num_workers = min(multiprocessing.cpu_count(), len(video_files))
    logging.info(f"🚀 Using {num_workers} parallel workers for frame extraction.")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(extract_frames, video_files)

if __name__ == "__main__":
    try:
        video_files = [
            (os.path.join(input_dir, f), os.path.join(output_dir, os.path.splitext(f)[0]))
            for f in os.listdir(input_dir) if f.endswith(".mp4")
        ]
    except Exception as e:
        logging.error(f"Error reading input directory: {e}")
        sys.exit(1)

    if video_files:
        logging.info(f"🚀 Extracting frames from {len(video_files)} videos in parallel...")
        process_videos_parallel(video_files)
    else:
        logging.info("✅ No videos found for processing.")
