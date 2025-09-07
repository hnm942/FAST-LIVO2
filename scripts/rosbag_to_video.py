import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import CompressedImage
import argparse
import os

def extract_video_from_rosbag(bag_file, topic, output_file, fps=30):
    """
    Extract video from rosbag compressed image topic
    """
    bridge = CvBridge()
    
    bag = rosbag.Bag(bag_file, 'r')
    
    info = bag.get_type_and_topic_info()
    if topic not in info.topics:
        print(f"Topic {topic} not found in bag file")
        print("Available topics:")
        for t in info.topics.keys():
            print(f"  - {t}")
        return False
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    frame_count = 0
    
    print(f"Processing topic: {topic}")
    print(f"Output file: {output_file}")
    
    try:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            try:
                if isinstance(msg, CompressedImage):
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                
                if out is None:
                    height, width, _ = cv_image.shape
                    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                    print(f"Video dimensions: {width}x{height}")
                
                out.write(cv_image)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
                
    except Exception as e:
        print(f"Error reading bag file: {e}")
        return False
    
    finally:
        bag.close()
        if out is not None:
            out.release()
    
    print(f"Video export completed! Total frames: {frame_count}")
    print(f"Output saved to: {output_file}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract video from ROS bag file')
    parser.add_argument('bag_file', help='Path to the ROS bag file')
    parser.add_argument('--topic', default='/camera/color/image_raw/compressed', 
                       help='Image topic to extract (default: /camera/color/image_raw/compressed)')
    parser.add_argument('--output', default='output_video.avi', 
                       help='Output video file (default: output_video.avi)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Output video FPS (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bag_file):
        print(f"Bag file not found: {args.bag_file}")
        exit(1)
    
    success = extract_video_from_rosbag(args.bag_file, args.topic, args.output, args.fps)
    
    if not success:
        exit(1)
