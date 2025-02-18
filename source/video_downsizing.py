#!/usr/bin/env python3
import cv2
import os
import argparse
import multiprocessing
import logging
import sys

def setup_logging(debug_mode=False):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Downsizing Videos")
    parser.add_argument("--input_dir", type=str, default="/media/mmlab/mldisk2/sign_language/out_3",
                        help="Path to the directory containing input video files")
    parser.add_argument("--output_dir", type=str, default="Data/video(256_256)",
                        help="Directory to save output videos")
    parser.add_argument("--debug", action="store_true", help="Save individual frames as images for debugging")
    return parser.parse_args()

args = parse_args()
setup_logging(args.debug)

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

def resize_and_center_crop(frame, target_size=256):
    """
    주어진 프레임을 종횡비를 유지하면서 짧은 변을 target_size 크기로 변환 후 긴 변을 중앙 크롭 수행
    """
    h, w, _ = frame.shape
    aspect_ratio = w / h  

    # 짧은 변을 target_size로 맞추고 비율 유지하여 리사이징
    if h < w:  # 가로가 긴 경우
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    else:  # 세로가 긴 경우
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    
    try:
        resized_frame = cv2.resize(frame, (new_w, new_h))
    except Exception as e:
        logging.error(f"Error resizing frame: {e}")
        raise

    # 중앙 크롭 수행
    crop_x = (new_w - target_size) // 2
    crop_y = (new_h - target_size) // 2
    cropped_frame = resized_frame[crop_y:crop_y+target_size, crop_x:crop_x+target_size]
    return cropped_frame

def process_video(input_video_path, output_video_path, debug=False):
    logging.info(f"🔄 Processing video: {input_video_path}")
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logging.error(f"❌ Cannot open video: {input_video_path}")
            return

        # FPS가 소수점일 수 있으므로 그대로 사용
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.warning(f"FPS not found or invalid for {input_video_path}. Defaulting to 30.")
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (256, 256))
    except Exception as e:
        logging.error(f"Initialization error for {input_video_path}: {e}")
        return

    frame_index = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = resize_and_center_crop(frame)
            
            if debug:
                debug_dir = os.path.join(output_dir, "debug_frames")
                os.makedirs(debug_dir, exist_ok=True)
                debug_frame_path = os.path.join(debug_dir, f"{os.path.basename(input_video_path)}_frame{frame_index:04d}.png")
                cv2.imwrite(debug_frame_path, processed_frame)
            
            out.write(processed_frame)
            frame_index += 1
        except Exception as e:
            logging.error(f"Error processing frame {frame_index} in {input_video_path}: {e}")
            break

    cap.release()
    out.release()
    logging.info(f"✅ Processed video saved at: {output_video_path}")

def process_video_wrapper(video_file):
    input_video_path, output_video_path = video_file
    process_video(input_video_path, output_video_path, debug=args.debug)

def process_videos_parallel(video_files):
    num_workers = min(multiprocessing.cpu_count(), len(video_files))
    logging.info(f"🚀 Using {num_workers} parallel workers for video processing.")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_video_wrapper, video_files)

if __name__ == "__main__":
    try:
        processed_files = {f.replace("_256.mp4", ".mp4") for f in os.listdir(output_dir) if f.endswith("_256.mp4")}
        all_files = {f for f in os.listdir(input_dir) if f.endswith(".mp4")}
    except Exception as e:
        logging.error(f"Error reading directories: {e}")
        sys.exit(1)
        
    pending_files = list(all_files - processed_files)
    logging.info(f"✅ Already processed files: {len(processed_files)}")
    logging.info(f"⏳ Pending files: {len(pending_files)}")

    video_files = [
        (os.path.join(input_dir, f), os.path.join(output_dir, f.replace(".mp4", "_256.mp4")))
        for f in pending_files
    ]

    if video_files:
        logging.info(f"🚀 Processing {len(video_files)} files in parallel...")
        process_videos_parallel(video_files)
    else:
        logging.info("✅ All files have been processed. No additional processing needed.")
