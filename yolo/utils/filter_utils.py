import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class FilterUtils:
    """
    Utility class for filtering detection results
    """
    
    @staticmethod
    def filter_by_classes(
        boxes: List[List[float]],
        scores: List[float],
        class_ids: List[int],
        target_classes: List[int]
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Filter detections to keep only specified target classes
        
        Args:
            boxes: List of bounding boxes in format [x1, y1, x2, y2]
            scores: List of confidence scores
            class_ids: List of class IDs
            target_classes: List of class IDs to keep
            
        Returns:
            Tuple of filtered boxes, scores, and class IDs
        """
        filtered_boxes = []
        filtered_scores = []
        filtered_class_ids = []
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id in target_classes:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_class_ids.append(class_id)
                
        return filtered_boxes, filtered_scores, filtered_class_ids
    
    @staticmethod
    def filter_by_confidence(
        boxes: List[List[float]],
        scores: List[float],
        class_ids: List[int],
        conf_threshold: float
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Filter detections by confidence score
        
        Args:
            boxes: List of bounding boxes in format [x1, y1, x2, y2]
            scores: List of confidence scores
            class_ids: List of class IDs
            conf_threshold: Confidence threshold
            
        Returns:
            Tuple of filtered boxes, scores, and class IDs
        """
        filtered_boxes = []
        filtered_scores = []
        filtered_class_ids = []
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= conf_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_class_ids.append(class_id)
                
        return filtered_boxes, filtered_scores, filtered_class_ids
    
    @staticmethod
    def filter_by_size(
        boxes: List[List[float]],
        scores: List[float],
        class_ids: List[int],
        image_shape: Tuple[int, int],
        min_area_ratio: float = 0.01
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Filter detections by object size relative to image
        
        Args:
            boxes: List of bounding boxes in format [x1, y1, x2, y2]
            scores: List of confidence scores
            class_ids: List of class IDs
            image_shape: Image shape as (height, width)
            min_area_ratio: Minimum area ratio to keep
            
        Returns:
            Tuple of filtered boxes, scores, and class IDs
        """
        image_area = image_shape[0] * image_shape[1]
        filtered_boxes = []
        filtered_scores = []
        filtered_class_ids = []
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / image_area
            
            if area_ratio >= min_area_ratio:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_class_ids.append(class_id)
                
        return filtered_boxes, filtered_scores, filtered_class_ids
    
    @staticmethod
    def is_static_object(
        box: List[float],
        class_id: int,
        static_classes: List[int] = [0, 2, 3],  # person, car, motorcycle
        min_area_ratio: float = 0.01,
        image_shape: Tuple[int, int] = None
    ) -> bool:
        """
        Determine if an object is likely to be static based on class and size
        
        Args:
            box: Bounding box in format [x1, y1, x2, y2]
            class_id: Class ID
            static_classes: List of class IDs considered potentially static
            min_area_ratio: Minimum area ratio to consider
            image_shape: Image shape as (height, width)
            
        Returns:
            Boolean indicating if object is likely static
        """
        # First check if class is in our list of potentially static objects
        if class_id not in static_classes:
            return False
            
        # If we have image shape, check size criteria
        if image_shape is not None:
            image_area = image_shape[0] * image_shape[1]
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / image_area
            
            # Small objects are less likely to be static obstacles
            if area_ratio < min_area_ratio:
                return False
                
        return True
