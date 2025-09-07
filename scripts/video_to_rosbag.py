#!/usr/bin/env python3

import rosbag
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CompressedImage
import argparse
import os
from datetime import datetime

def video_to_rosbag(video_file, output_bag, topic_name="/camera/color/image_raw", compressed=True, fps=None):
    """
    Convert video file to ROS bag
    """
    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
        return False
    
    # Initialize CV bridge
    bridge = CvBridge()
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening video file: {video_file}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use video FPS if not specified
    if fps is None:
        fps = video_fps
    
    print(f"Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Original FPS: {video_fps}")
    print(f"  - Output FPS: {fps}")
    print(f"  - Compressed: {compressed}")
    
    # Create rosbag
    bag = rosbag.Bag(output_bag, 'w')
    
    # Calculate time step between frames
    time_step = rospy.Duration(1.0 / fps)
    start_time = rospy.Time.now()
    current_time = start_time
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB (ROS standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if compressed:
                # Create compressed image message
                msg = CompressedImage()
                msg.header.stamp = current_time
                msg.header.frame_id = "camera"
                msg.format = "jpeg"
                
                # Encode frame as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                if result:
                    msg.data = encimg.tobytes()
                else:
                    print(f"Failed to encode frame {frame_count}")
                    continue
                
                # Write to bag with compressed topic name
                compressed_topic = topic_name + "/compressed"
                bag.write(compressed_topic, msg, current_time)
            else:
                # Create raw image message
                try:
                    msg = bridge.cv2_to_imgmsg(frame_rgb, "rgb8")
                    msg.header.stamp = current_time
                    msg.header.frame_id = "camera"
                    
                    # Write to bag
                    bag.write(topic_name, msg, current_time)
                except Exception as e:
                    print(f"Error converting frame {frame_count}: {e}")
                    continue
            
            # Update time and frame count
            current_time += time_step
            frame_count += 1
            
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nConversion interrupted by user")
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False
    
    finally:
        cap.release()
        bag.close()
    
    print(f"\nConversion completed!")
    print(f"  - Processed frames: {frame_count}")
    print(f"  - Output bag: {output_bag}")
    print(f"  - Topic: {compressed_topic if compressed else topic_name}")
    
    return True

def add_additional_topics(bag_file, imu_topic="/imu/data", lidar_topic="/velodyne_points"):
    """
    Add placeholder IMU and LiDAR topics for SLAM compatibility
    """
    print(f"Adding placeholder topics to {bag_file}...")
    
    # This is a placeholder - in real scenarios you'd need actual sensor data
    # For now, we just document what topics would be needed
    topics_info = {
        imu_topic: "sensor_msgs/Imu - IMU data needed for SLAM",
        lidar_topic: "sensor_msgs/PointCloud2 - LiDAR data needed for SLAM",
        "/tf": "tf2_msgs/TFMessage - Transform data",
        "/tf_static": "tf2_msgs/TFMessage - Static transforms"
    }
    
    print("Note: For FAST-LIVO2 SLAM, you'll also need:")
    for topic, description in topics_info.items():
        print(f"  - {topic}: {description}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert video file to ROS bag')
    parser.add_argument('video_file', help='Path to input video file')
    parser.add_argument('--output', '-o', default='output.bag', 
                       help='Output bag file name (default: output.bag)')
    parser.add_argument('--topic', default='/camera/color/image_raw',
                       help='Image topic name (default: /camera/color/image_raw)')
    parser.add_argument('--fps', type=float, default=None,
                       help='Output FPS (default: use video FPS)')
    parser.add_argument('--raw', action='store_true',
                       help='Save as raw images instead of compressed (default: compressed)')
    parser.add_argument('--add-slam-topics', action='store_true',
                       help='Show info about additional topics needed for SLAM')
    
    args = parser.parse_args()
    
    # Convert video to bag
    success = video_to_rosbag(
        video_file=args.video_file,
        output_bag=args.output,
        topic_name=args.topic,
        compressed=not args.raw,
        fps=args.fps
    )
    
    if success and args.add_slam_topics:
        add_additional_topics(args.output)
    
    if not success:
        exit(1)
    
    print(f"\nTo play the bag file:")
    print(f"  rosbag play {args.output}")
    print(f"\nTo view images:")
    if args.raw:
        print(f"  rosrun image_view image_view image:={args.topic}")
    else:
        print(f"  rosrun image_view image_view image:={args.topic} _image_transport:=compressed")
