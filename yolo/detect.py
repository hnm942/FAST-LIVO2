#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project utilities
from utils.visualizer import Visualizer
from utils.filter_utils import FilterUtils


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--source', type=str, default='data/images',
                        help='source directory of images or videos')
    parser.add_argument('--weights', type=str, default='weights/yolov8n.pt',
                        help='path to model weights')
    parser.add_argument('--config', type=str, default='configs/yolo_v12_config.yaml',
                        help='path to configuration file')
    parser.add_argument('--output', type=str, default='results/outputs',
                        help='output directory')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='object confidence threshold (overrides config)')
    parser.add_argument('--iou-thres', type=float, default=None,
                        help='IOU threshold for NMS (overrides config)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video detection results')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for output video')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0 2 3 for person, car, motorcycle')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_image(model, img_path, conf_thres, iou_thres, target_classes, args):
    """Process a single image with the model"""
    # Get filename without extension
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Run inference
    results = model(img_path, conf=conf_thres, iou=iou_thres)
    
    # Process results
    for i, result in enumerate(results):
        # Get detections
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by target classes if specified
        if target_classes:
            boxes, scores, class_ids = FilterUtils.filter_by_classes(
                boxes, scores, class_ids, target_classes
            )
        
        # Load original image for visualization
        img = cv2.imread(img_path)
        
        # Draw detections
        output_img = Visualizer.draw_detections(
            img, boxes, scores, class_ids, 
            filter_classes=target_classes
        )
        
        # Save output image
        output_path = os.path.join(args.output, f"{base_filename}_det.jpg")
        Visualizer.save_image(output_img, output_path)
        
        # Save detection results as text if requested
        if args.save_txt:
            txt_path = os.path.join(args.output, f"{base_filename}.txt")
            with open(txt_path, 'w') as f:
                for box, score, class_id in zip(boxes, scores, class_ids):
                    # Convert box to YOLO format (x_center, y_center, width, height)
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2 / img.shape[1]
                    y_center = (y1 + y2) / 2 / img.shape[0]
                    width = (x2 - x1) / img.shape[1]
                    height = (y2 - y1) / img.shape[0]
                    
                    # Write to file
                    f.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")
        
        # Display image if requested
        if args.view_img:
            cv2.imshow(f"Detection - {filename}", output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow quitting with 'q'
                break
                
        return output_img, boxes, scores, class_ids


def process_video(model, video_path, conf_thres, iou_thres, target_classes, args):
    """Process a video with the model"""
    # Get filename without extension
    filename = os.path.basename(video_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.fps is None else args.fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Create video writer if saving is requested
    output_video = None
    if args.save_vid:
        output_path = os.path.join(args.output, f"{base_filename}_det.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    with tqdm(total=total_frames, desc=f"Processing {filename}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on frame
            results = model(frame, conf=conf_thres, iou=iou_thres)
            
            # Process results
            for i, result in enumerate(results):
                # Get detections
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter by target classes if specified
                if target_classes:
                    boxes, scores, class_ids = FilterUtils.filter_by_classes(
                        boxes, scores, class_ids, target_classes
                    )
                
                # Draw detections
                output_frame = Visualizer.draw_detections(
                    frame, boxes, scores, class_ids, 
                    filter_classes=target_classes
                )
                
                # Save frame if requested
                if args.save_txt and frame_count % 30 == 0:  # Save every 30 frames
                    txt_path = os.path.join(args.output, f"{base_filename}_frame_{frame_count:06d}.txt")
                    with open(txt_path, 'w') as f:
                        for box, score, class_id in zip(boxes, scores, class_ids):
                            # Convert box to YOLO format
                            x1, y1, x2, y2 = box
                            x_center = (x1 + x2) / 2 / frame.shape[1]
                            y_center = (y1 + y2) / 2 / frame.shape[0]
                            width = (x2 - x1) / frame.shape[1]
                            height = (y2 - y1) / frame.shape[0]
                            
                            # Write to file
                            f.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")
                
                # Write frame to output video
                if output_video is not None:
                    output_video.write(output_frame)
                
                # Display frame if requested
                if args.view_img:
                    cv2.imshow(f"Detection - {filename}", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow quitting with 'q'
                        break
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    if output_video is not None:
        output_video.release()
    
    print(f"Video processing complete. Processed {frame_count} frames.")
    if args.save_vid:
        print(f"Output saved to {output_path}")


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set detection parameters from config or command line arguments
    conf_thres = args.conf_thres if args.conf_thres is not None else config['detection']['conf_threshold']
    iou_thres = args.iou_thres if args.iou_thres is not None else config['detection']['iou_threshold']
    
    # Set target classes from config or command line arguments
    target_classes = args.classes if args.classes is not None else config['classes']
    
    # Initialize model
    print(f"Loading YOLO-seg model from {args.weights}...")
    try:
        # Try to load with Ultralytics YOLO
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have the correct weights file and it's compatible with YOLOv8.")
        sys.exit(1)
    
    # Determine file type (image or video)
    source_path = Path(args.source)
    
    # File extensions
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    if source_path.is_dir():
        # Get all image and video files in the directory
        img_files = [
            str(p) for p in source_path.glob('**/*') 
            if p.suffix.lower() in img_extensions
        ]
        
        video_files = [
            str(p) for p in source_path.glob('**/*') 
            if p.suffix.lower() in video_extensions
        ]
        
        # Process images
        if img_files:
            print(f"Found {len(img_files)} images. Processing...")
            for img_path in tqdm(img_files):
                process_image(model, img_path, conf_thres, iou_thres, target_classes, args)
        
        # Process videos
        if video_files:
            print(f"Found {len(video_files)} videos. Processing...")
            for video_path in video_files:
                process_video(model, video_path, conf_thres, iou_thres, target_classes, args)
                
        if not img_files and not video_files:
            print(f"No images or videos found in {args.source}")
            sys.exit(1)
    else:
        # Process a single file
        if source_path.suffix.lower() in img_extensions:
            print(f"Processing image: {args.source}")
            process_image(model, str(source_path), conf_thres, iou_thres, target_classes, args)
        elif source_path.suffix.lower() in video_extensions:
            print(f"Processing video: {args.source}")
            process_video(model, str(source_path), conf_thres, iou_thres, target_classes, args)
        else:
            print(f"Unsupported file type: {source_path.suffix}")
            sys.exit(1)
    
    # Clean up
    if args.view_img:
        cv2.destroyAllWindows()
    
    print(f"Detection complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
