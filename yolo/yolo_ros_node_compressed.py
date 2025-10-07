#!/usr/bin/env python3
"""
ROS 1 Node for YOLO Segmentation with Double View
Supports both raw and compressed image topics
"""

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import time


class YOLOSegmentationNode:
    def __init__(self):
        """Initialize the YOLO Segmentation ROS node"""
        rospy.init_node('yolo_segmentation_node', anonymous=True)
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Parameters
        default_weights = os.path.join(script_dir, 'weights/yolo11n-seg.pt')
        self.weights_path = rospy.get_param('~weights', default_weights)
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.25)
        self.iou_threshold = rospy.get_param('~iou_threshold', 0.45)
        self.alpha = rospy.get_param('~alpha', 0.5)
        self.device = rospy.get_param('~device', '')
        self.person_class_id = rospy.get_param('~person_class_id', 0)
        self.use_compressed = rospy.get_param('~use_compressed', False)
        
        # Topics
        image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        segmented_topic = rospy.get_param('~segmented_topic', '/yolo/segmented_image')
        walkability_topic = rospy.get_param('~walkability_topic', '/yolo/walkability_mask')
        double_view_topic = rospy.get_param('~double_view_topic', '/yolo/double_view')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Check if weights file exists
        if not os.path.exists(self.weights_path):
            rospy.logerr(f"Weights file not found: {self.weights_path}")
            rospy.signal_shutdown("Weights file not found")
            return
        
        # Load YOLO model
        rospy.loginfo(f"Loading YOLO segmentation model from {self.weights_path}...")
        try:
            self.model = YOLO(self.weights_path)
            test_img = np.zeros((320, 320, 3), dtype=np.uint8)
            test_results = self.model(test_img, conf=0.5, verbose=False)
            rospy.loginfo("Model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            rospy.signal_shutdown("Failed to load YOLO model")
            return
        
        # Color map
        self.color_map = {
            0: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0),
            5: (255, 255, 0), 7: (255, 0, 255), 1: (0, 255, 255),
        }
        
        # COCO classes
        self.class_names = [
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
        
        # Publishers
        self.segmented_pub = rospy.Publisher(segmented_topic, Image, queue_size=1)
        self.walkability_pub = rospy.Publisher(walkability_topic, Image, queue_size=1)
        self.double_view_pub = rospy.Publisher(double_view_topic, Image, queue_size=1)
        
        # Subscriber - auto-detect compressed or raw
        if self.use_compressed or '/compressed' in image_topic:
            if not '/compressed' in image_topic:
                image_topic = image_topic + '/compressed'
            rospy.loginfo(f"Subscribing to COMPRESSED image: {image_topic}")
            self.image_sub = rospy.Subscriber(image_topic, CompressedImage, 
                                            self.compressed_callback, queue_size=1, buff_size=2**24)
        else:
            rospy.loginfo(f"Subscribing to RAW image: {image_topic}")
            self.image_sub = rospy.Subscriber(image_topic, Image, 
                                            self.image_callback, queue_size=1, buff_size=2**24)
        
        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        rospy.loginfo("="*50)
        rospy.loginfo("YOLO Segmentation Node Ready!")
        rospy.loginfo(f"Waiting for images on: {image_topic}")
        rospy.loginfo("="*50)
    
    def compressed_callback(self, msg):
        """Callback for compressed images"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                rospy.logerr("Failed to decode compressed image")
                return
            
            # Create a fake header for compatibility
            class FakeMsg:
                def __init__(self):
                    self.header = msg.header
            
            fake_msg = FakeMsg()
            self.process_image(cv_image, fake_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in compressed callback: {e}")
    
    def image_callback(self, msg):
        """Callback for raw images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_image(cv_image, msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
    
    def process_image(self, cv_image, msg):
        """Process image with YOLO"""
        h, w = cv_image.shape[:2]
        
        # Inference
        start_time = time.time()
        try:
            results = self.model(cv_image, conf=self.conf_threshold, 
                               iou=self.iou_threshold, verbose=False)
            inference_time = time.time() - start_time
            
            self.frame_count += 1
            self.total_inference_time += inference_time
            
            if self.frame_count % 30 == 0:
                fps = 1.0 / inference_time if inference_time > 0 else 0
                avg_time = self.total_inference_time / self.frame_count
                rospy.loginfo(f"Frame {self.frame_count}: {inference_time:.4f}s, "
                            f"FPS: {fps:.2f}, Avg: {avg_time:.4f}s")
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
            return
        
        # Create segmented image
        output_img = cv_image.copy()
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            if hasattr(result, 'masks') and result.masks is not None:
                for j, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                    color = self.color_map.get(class_id, (255, 0, 0))
                    
                    if j < len(result.masks):
                        mask = result.masks[j].data.cpu().numpy()
                        mask = cv2.resize(mask[0].astype(np.uint8), (w, h), 
                                        interpolation=cv2.INTER_NEAREST)
                        
                        # Apply mask
                        overlay = output_img.copy()
                        overlay[mask > 0] = color
                        cv2.addWeighted(overlay, self.alpha, output_img, 1 - self.alpha, 0, output_img)
                    
                    # Draw box and label
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {score:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(output_img, (x1, y1-lh-10), (x1+lw, y1), color, -1)
                    cv2.putText(output_img, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Generate walkability mask
        walkability_mask = np.ones((h, w), dtype=np.uint8)
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for j, class_id in enumerate(class_ids):
                    if class_id == self.person_class_id and j < len(result.masks):
                        mask = result.masks[j].data.cpu().numpy()
                        mask = cv2.resize(mask[0].astype(np.uint8), (w, h), 
                                        interpolation=cv2.INTER_NEAREST)
                        walkability_mask[mask > 0] = 0
        
        # Create double view
        walkability_vis = cv2.cvtColor(walkability_mask * 255, cv2.COLOR_GRAY2BGR)
        double_view = np.hstack([cv_image, walkability_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(double_view, (10, 10), (150, 50), (0, 0, 0), -1)
        cv2.putText(double_view, "Original", (15, 40), font, 1.0, (255, 255, 255), 2)
        cv2.rectangle(double_view, (w+10, 10), (w+250, 50), (0, 0, 0), -1)
        cv2.putText(double_view, "Walkability", (w+15, 40), font, 1.0, (255, 255, 255), 2)
        
        # Publish
        try:
            seg_msg = self.bridge.cv2_to_imgmsg(output_img, encoding='bgr8')
            seg_msg.header = msg.header
            self.segmented_pub.publish(seg_msg)
            
            walk_msg = self.bridge.cv2_to_imgmsg(walkability_mask * 255, encoding='mono8')
            walk_msg.header = msg.header
            self.walkability_pub.publish(walk_msg)
            
            double_msg = self.bridge.cv2_to_imgmsg(double_view, encoding='bgr8')
            double_msg.header = msg.header
            self.double_view_pub.publish(double_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Publishing error: {e}")
    
    def run(self):
        rospy.spin()


def main():
    try:
        node = YOLOSegmentationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
