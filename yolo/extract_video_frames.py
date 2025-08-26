#!/usr/bin/env python3
import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, num_frames=10):
    """Extract frames from a video file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Calculate frame interval
    if num_frames >= total_frames:
        interval = 1
        num_frames = total_frames
    else:
        interval = total_frames // num_frames
    
    # Extract frames
    frames = []
    for i in range(num_frames):
        # Set position
        frame_pos = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame at position {frame_pos}")
            continue
            
        # Save frame
        frame_file = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(frame_file, frame)
        frames.append(frame_file)
        print(f"Saved {frame_file}")
    
    # Release video
    cap.release()
    return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", default="data/videos/roads.mp4", help="Path to video file")
    parser.add_argument("--output", default="data/images/frames", help="Output directory")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to extract")
    args = parser.parse_args()
    
    # Extract frames
    extracted_frames = extract_frames(args.video, args.output, args.frames)
    print(f"Extracted {len(extracted_frames)} frames to {args.output}")
