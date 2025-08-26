#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project utilities
from utils.visualizer import Visualizer
from utils.filter_utils import FilterUtils


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Filter Static Objects from YOLOv8 Detections')
    parser.add_argument('--source', type=str, default='results/outputs',
                        help='source directory of detection images')
    parser.add_argument('--output', type=str, default='results/filtered',
                        help='output directory for filtered results')
    parser.add_argument('--config', type=str, default='configs/yolo_v12_config.yaml',
                        help='path to configuration file')
    parser.add_argument('--min-area', type=float, default=None,
                        help='minimum area ratio to consider (overrides config)')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0 2 3 for person, car, motorcycle')
    parser.add_argument('--txt-source', type=str, default=None,
                        help='source directory of detection text files (if available)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_detection_txt(txt_path):
    """Parse YOLO format detection text file"""
    boxes = []
    scores = []
    class_ids = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # class_id, x_center, y_center, width, height, confidence
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                score = float(parts[5])
                
                # Convert from YOLO format to [x1, y1, x2, y2]
                # Note: These are normalized coordinates (0-1)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)
    
    return boxes, scores, class_ids


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set parameters from config or command line arguments
    target_classes = args.classes if args.classes is not None else config['static_filter']['target_classes']
    min_area_ratio = args.min_area if args.min_area is not None else config['static_filter']['min_area_ratio']
    
    # Get image files
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    source_path = Path(args.source)
    
    if source_path.is_dir():
        img_files = [
            str(p) for p in source_path.glob('**/*') 
            if p.suffix.lower() in img_extensions
        ]
    else:
        img_files = [str(source_path)]
    
    if not img_files:
        print(f"No images found in {args.source}")
        sys.exit(1)
    
    print(f"Found {len(img_files)} images. Processing...")
    
    # Process each image
    for img_path in tqdm(img_files):
        # Get filename without extension
        filename = os.path.basename(img_path)
        base_filename = os.path.splitext(filename)[0]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Check if we have corresponding detection text file
        if args.txt_source:
            txt_path = os.path.join(args.txt_source, f"{base_filename.replace('_det', '')}.txt")
            if os.path.exists(txt_path):
                # Parse detection text file
                boxes, scores, class_ids = parse_detection_txt(txt_path)
                
                # Convert normalized coordinates to absolute
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    boxes[i] = [
                        x1 * img_width,
                        y1 * img_height,
                        x2 * img_width,
                        y2 * img_height
                    ]
            else:
                print(f"Warning: No detection text file found for {filename}")
                continue
        else:
            # If no text file, we need to extract detections from the image
            # This is a placeholder - in a real implementation, you would need
            # to use object detection here or have a way to parse the detection image
            print(f"Warning: No detection text source provided. Skipping {filename}")
            continue
        
        # Filter by target classes
        boxes, scores, class_ids = FilterUtils.filter_by_classes(
            boxes, scores, class_ids, target_classes
        )
        
        # Filter by size
        boxes, scores, class_ids = FilterUtils.filter_by_size(
            boxes, scores, class_ids, (img_height, img_width), min_area_ratio
        )
        
        # Draw filtered detections
        output_img = Visualizer.draw_detections(
            img, boxes, scores, class_ids, 
            filter_classes=target_classes
        )
        
        # Save output image
        output_path = os.path.join(args.output, f"{base_filename}_static.jpg")
        Visualizer.save_image(output_img, output_path)
        
        # Save filtered detection results as text
        txt_output_path = os.path.join(args.output, f"{base_filename}_static.txt")
        with open(txt_output_path, 'w') as f:
            for box, score, class_id in zip(boxes, scores, class_ids):
                # Convert box to YOLO format (x_center, y_center, width, height)
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Write to file
                f.write(f"{class_id} {x_center} {y_center} {width} {height} {score}\n")
    
    print(f"Filtering complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
