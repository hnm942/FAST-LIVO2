import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


class Visualizer:
    """
    Utility class for visualizing detection results
    """
    # COCO class names for reference
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Colors for visualization (BGR format for OpenCV)
    COLORS = {
        'person': (0, 0, 255),      # Red for people
        'car': (0, 255, 0),         # Green for cars
        'motorcycle': (255, 0, 0),  # Blue for motorcycles
        'default': (255, 255, 0)    # Cyan for other classes
    }
    
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        boxes: List[List[float]],
        scores: List[float],
        class_ids: List[int],
        class_names: Optional[List[str]] = None,
        filter_classes: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on an image
        
        Args:
            image: Input image as numpy array (BGR format)
            boxes: List of bounding boxes in format [x1, y1, x2, y2]
            scores: List of confidence scores
            class_ids: List of class IDs
            class_names: Optional list of class names
            filter_classes: Optional list of class IDs to display
            
        Returns:
            Image with drawn detections
        """
        if class_names is None:
            class_names = Visualizer.COCO_CLASSES
            
        img_copy = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Skip if we're filtering and this class is not in the filter
            if filter_classes is not None and class_id not in filter_classes:
                continue
                
            # Get class name
            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
                
            # Get color for this class
            color = Visualizer.COLORS.get(class_name, Visualizer.COLORS['default'])
            
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and score
            label = f"{class_name}: {score:.2f}"
            
            # Get text size and baseline
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                img_copy, 
                (x1, y1 - label_height - baseline - 5), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_copy,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
        return img_copy
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> None:
        """
        Save image to disk
        
        Args:
            image: Image to save
            output_path: Path to save the image
        """
        cv2.imwrite(output_path, image)
        
    @staticmethod
    def show_image(image: np.ndarray, title: str = "Detection Result") -> None:
        """
        Display image using matplotlib
        
        Args:
            image: Image to display (BGR format)
            title: Window title
        """
        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
