#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
import time
import subprocess
import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project utilities
from utils.visualizer import Visualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation')
    parser.add_argument('--source', type=str, default='data/images',
                        help='source directory of images or videos')
    parser.add_argument('--weights', type=str, default='weights/yolov8n-seg.pt',
                        help='path to model weights')
    parser.add_argument('--config', type=str, default='configs/yolo_v12_config.yaml',
                        help='path to configuration file')
    parser.add_argument('--output', type=str, default='results/segmentation',
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
                        help='save video segmentation results')
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
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='mask transparency (0-1)')
    parser.add_argument('--export-excel', action='store_true',
                        help='export inference time results to Excel')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_color_map():
    """Create a vibrant color map for better visualization"""
    return {
        0: (0, 0, 255),      # person - red
        2: (0, 255, 0),      # car - green
        3: (255, 0, 0),      # motorcycle - blue
        5: (255, 255, 0),    # bus - cyan
        7: (255, 0, 255),    # truck - magenta
        1: (0, 255, 255),    # bicycle - yellow
        9: (128, 0, 255),    # traffic light - purple
        10: (255, 128, 0),   # fire hydrant - orange
        11: (0, 255, 128),   # stop sign - teal
        12: (128, 128, 255), # parking meter - light blue
    }


def apply_segmentation_mask(image, mask, color, alpha=0.5):
    """Apply a segmentation mask to an image with proper color and transparency
    
    Args:
        image: Original image
        mask: Binary mask (should be same size as image)
        color: Color tuple (B, G, R)
        alpha: Transparency value (0-1)
        
    Returns:
        Image with applied mask
    """
    # Create a copy of the image
    result = image.copy()
    
    # Create a colored overlay
    overlay = result.copy()
    overlay[mask > 0] = color
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result


def process_image(model, img_path, conf_thres, iou_thres, target_classes, args, inference_data=None):
    """Process a single image with the segmentation model"""
    # Get filename without extension
    filename = os.path.basename(img_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Measure inference time
    start_time = time.time()
    results = model(img_path, conf=conf_thres, iou=iou_thres)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Store inference data if requested
    if inference_data is not None:
        inference_data.append({
            'file_name': os.path.basename(img_path),
            'file_type': 'image',
            'resolution': f"{w}x{h}",
            'inference_time': inference_time,
            'num_objects': sum(len(result.boxes) for result in results),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Create a copy of the image for visualization
    output_img = img.copy()
    
    # Create color map
    color_map = create_color_map()
    
    # Process results
    for i, result in enumerate(results):
        # Get detections
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by target classes if specified
        if target_classes:
            indices = [i for i, cls in enumerate(class_ids) if cls in target_classes]
            if indices:
                boxes = boxes[indices]
                scores = scores[indices]
                class_ids = class_ids[indices]
            else:
                # No objects of target classes found
                continue
        
        # Get segmentation masks if available
        if hasattr(result, 'masks') and result.masks is not None:
            # Process each detection
            for j, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                # Get class name
                if 0 <= class_id < len(Visualizer.COCO_CLASSES):
                    class_name = Visualizer.COCO_CLASSES[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                # Get color for this class
                if class_id in color_map:
                    color = color_map[class_id]
                else:
                    color = Visualizer.COLORS.get(class_name, Visualizer.COLORS['default'])
                
                # Get mask for this detection
                if j < len(result.masks):
                    # Get the mask
                    mask = result.masks[j].data.cpu().numpy()
                    
                    # Resize mask to match image dimensions
                    mask = cv2.resize(mask[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Apply the mask
                    output_img = apply_segmentation_mask(output_img, mask, color, args.alpha)
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name and score
                label = f"{class_name}: {score:.2f}"
                
                # Get text size and baseline
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    output_img, 
                    (x1, y1 - label_height - baseline - 5), 
                    (x1 + label_width, y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output_img,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        else:
            print(f"No segmentation masks found for {filename}")
    
    # Save output image
    output_path = os.path.join(args.output, f"{base_filename}_seg.jpg")
    cv2.imwrite(output_path, output_img)
    
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
        cv2.imshow(f"Segmentation - {filename}", output_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Wait for 'q' to close
            cv2.destroyAllWindows()
    
    return output_img


def process_video(model, video_path, conf_thres, iou_thres, target_classes, args, inference_data=None):
    """Process a video with the segmentation model"""
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
    output_path = None
    if args.save_vid:
        output_path = os.path.join(args.output, f"{base_filename}_seg.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create color map
    color_map = create_color_map()
    
    # Process frames
    frame_count = 0
    with tqdm(total=total_frames, desc=f"Processing {filename}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Measure inference time for this frame
            start_time = time.time()
            results = model(frame, conf=conf_thres, iou=iou_thres)
            frame_inference_time = time.time() - start_time
            
            # Store inference data if requested (sample every 10 frames to avoid too much data)
            if inference_data is not None and frame_count % 10 == 0:
                inference_data.append({
                    'file_name': f"{os.path.basename(video_path)}_frame_{frame_count}",
                    'file_type': 'video_frame',
                    'resolution': f"{w}x{h}",
                    'inference_time': frame_inference_time,
                    'num_objects': sum(len(result.boxes) for result in results),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Create a copy of the frame for visualization
            output_frame = frame.copy()
            
            # Process results
            for i, result in enumerate(results):
                # Get detections
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter by target classes if specified
                if target_classes:
                    indices = [i for i, cls in enumerate(class_ids) if cls in target_classes]
                    if indices:
                        boxes = boxes[indices]
                        scores = scores[indices]
                        class_ids = class_ids[indices]
                    else:
                        # No objects of target classes found
                        continue
                
                # Get segmentation masks if available
                if hasattr(result, 'masks') and result.masks is not None:
                    # Process each detection
                    for j, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                        # Get class name
                        if 0 <= class_id < len(Visualizer.COCO_CLASSES):
                            class_name = Visualizer.COCO_CLASSES[class_id]
                        else:
                            class_name = f"Class {class_id}"
                        
                        # Get color for this class
                        if class_id in color_map:
                            color = color_map[class_id]
                        else:
                            color = Visualizer.COLORS.get(class_name, Visualizer.COLORS['default'])
                        
                        # Get mask for this detection
                        if j < len(result.masks):
                            # Get the mask
                            mask = result.masks[j].data.cpu().numpy()
                            
                            # Resize mask to match frame dimensions
                            mask = cv2.resize(mask[0].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            
                            # Apply the mask
                            output_frame = apply_segmentation_mask(output_frame, mask, color, args.alpha)
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label with class name and score
                        label = f"{class_name}: {score:.2f}"
                        
                        # Get text size and baseline
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            output_frame, 
                            (x1, y1 - label_height - baseline - 5), 
                            (x1 + label_width, y1), 
                            color, 
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            output_frame,
                            label,
                            (x1, y1 - baseline - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
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
                cv2.imshow(f"Segmentation - {filename}", output_frame)
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
    
    # Automatically open the video file if it was saved
    if args.save_vid and output_path and os.path.exists(output_path):
        print(f"Opening video: {output_path}")
        try:
            # Use the default video player on macOS
            subprocess.Popen(['open', output_path])
        except Exception as e:
            print(f"Error opening video: {e}")


def export_inference_data_to_excel(inference_data, output_dir):
    """Export inference time data to Excel file"""
    if not inference_data:
        print("No inference data to export")
        return
    
    # Create DataFrame from inference data
    df = pd.DataFrame(inference_data)
    
    # Create Excel file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_dir, f"inference_results_{timestamp}.xlsx")
    
    # Export to Excel
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Inference data exported to {excel_path}")
    
    # Calculate and add summary statistics
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        # Calculate statistics by file type
        stats = df.groupby('file_type').agg({
            'inference_time': ['mean', 'min', 'max', 'std'],
            'num_objects': ['mean', 'min', 'max', 'sum']
        })
        stats.to_excel(writer, sheet_name='Summary')
    
    return excel_path


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Always enable view_img by default
    args.view_img = True
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set detection parameters from config or command line arguments
    conf_thres = args.conf_thres if args.conf_thres is not None else config['detection']['conf_threshold']
    iou_thres = args.iou_thres if args.iou_thres is not None else config['detection']['iou_threshold']
    
    # Set target classes from config or command line arguments
    target_classes = args.classes if args.classes is not None else config['classes']
    
    # Initialize list to store inference data if Excel export is requested
    inference_data = [] if args.export_excel else None
    
    # Initialize model
    print(f"Loading YOLOv8 segmentation model from {args.weights}...")
    try:
        # Try to load with Ultralytics YOLO
        model = YOLO(args.weights)
        
        # Check if it's a segmentation model
        if not any(task in str(model.task) for task in ['segment', 'seg']):
            print(f"Warning: The model {args.weights} may not be a segmentation model.")
            print("Make sure you're using a YOLOv8 model with the '-seg' suffix.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have the correct weights file and it's compatible with YOLOv8 segmentation.")
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
                process_image(model, img_path, conf_thres, iou_thres, target_classes, args, inference_data)
        
        # Process videos
        if video_files:
            print(f"Found {len(video_files)} videos. Processing...")
            for video_path in video_files:
                process_video(model, video_path, conf_thres, iou_thres, target_classes, args, inference_data)
                
        if not img_files and not video_files:
            print(f"No images or videos found in {args.source}")
            sys.exit(1)
    else:
        # Process a single file
        if source_path.suffix.lower() in img_extensions:
            print(f"Processing image: {args.source}")
            process_image(model, str(source_path), conf_thres, iou_thres, target_classes, args, inference_data)
        elif source_path.suffix.lower() in video_extensions:
            print(f"Processing video: {args.source}")
            process_video(model, str(source_path), conf_thres, iou_thres, target_classes, args, inference_data)
        else:
            print(f"Unsupported file type: {source_path.suffix}")
            sys.exit(1)
    
    # Clean up
    if args.view_img:
        cv2.destroyAllWindows()
    
    # Export inference data to Excel if requested
    if args.export_excel and inference_data:
        excel_path = export_inference_data_to_excel(inference_data, args.output)
        print(f"Inference time data exported to {excel_path}")
    
    print(f"Segmentation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
