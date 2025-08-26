#!/usr/bin/env python3
import os
import sys
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract frames from videos for YOLO detection')
    parser.add_argument('--source', type=str, default='data/videos',
                        help='source directory of videos')
    parser.add_argument('--output', type=str, default='data/images/frames',
                        help='output directory for extracted frames')
    parser.add_argument('--num-frames', type=int, default=10,
                        help='number of frames to extract from each video')
    parser.add_argument('--detect', action='store_true',
                        help='run detection on extracted frames')
    return parser.parse_args()


def extract_frames(video_path, output_dir, num_frames=10):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        
    Returns:
        List of paths to extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Calculate frame interval to extract evenly spaced frames
    if total_frames <= num_frames:
        # If video has fewer frames than requested, extract all frames
        frame_interval = 1
        num_frames = total_frames
    else:
        # Otherwise, extract evenly spaced frames
        frame_interval = total_frames // num_frames
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Extract frames
    extracted_frames = []
    for i in tqdm(range(num_frames), desc=f"Extracting frames from {video_name}"):
        # Set frame position
        frame_pos = i * frame_interval
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # Read frame
        ret, frame = video.read()
        if not ret:
            print(f"Error: Could not read frame at position {frame_pos}")
            continue
        
        # Save frame - use a simple naming convention directly in output_dir
        frame_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)
    
    # Release video
    video.release()
    
    return extracted_frames


def run_detection(frame_paths):
    """
    Run YOLO detection on extracted frames
    
    Args:
        frame_paths: List of paths to extracted frames
    """
    # Check if frames exist
    existing_frames = [path for path in frame_paths if os.path.exists(path)]
    if not existing_frames:
        print("Error: No extracted frames found")
        return
        
    print(f"Running detection on {len(existing_frames)} frames...")
    
    # Run detection using subprocess instead of importing
    cmd = [
        'python', 'detect.py',
        '--source', ' '.join(existing_frames),
        '--weights', 'weights/yolov8n.pt',
        '--conf-thres', '0.25'
    ]
    
    import subprocess
    subprocess.run(cmd)


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    source_path = Path(args.source)
    
    if source_path.is_dir():
        video_files = [
            str(p) for p in source_path.glob('**/*') 
            if p.suffix.lower() in video_extensions
        ]
    else:
        video_files = [str(source_path)]
    
    if not video_files:
        print(f"No videos found in {args.source}")
        sys.exit(1)
    
    print(f"Found {len(video_files)} videos. Processing...")
    
    # Create a single output directory for all frames
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames from each video
    all_frames = []
    for video_path in video_files:
        # Extract frames directly to the output directory
        frames = extract_frames(video_path, output_dir, args.num_frames)
        all_frames.extend(frames)
    
    print(f"Extracted {len(all_frames)} frames from {len(video_files)} videos")
    print(f"Frames saved to: {output_dir}")
    
    # Run detection if requested
    if args.detect and all_frames:
        run_detection(all_frames)


if __name__ == "__main__":
    main()
