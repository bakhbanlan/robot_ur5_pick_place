#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import datetime
from collections import defaultdict
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs
from ultralytics import YOLO
from tf.transformations import quaternion_from_euler

# --- ฟังก์ชันแปลงพิกัดจาก camera frame ไป robot base frame ---
def transform_to_robot_frame(x_cam, y_cam, z_cam):
    # สำหรับ Realsense D435i:
    #   x_cam: ขวา, y_cam: ลง, z_cam: หน้า
    # สำหรับ UR5e (ระบบ robot base):
    #   x_robot: หน้า, y_robot: ซ้าย, z_robot: ขึ้น
    # ดังนั้นให้แปลงดังนี้:
    x_robot = z_cam      # ระยะหน้าของหุ่นยนต์ = ระยะจากกล้องไปข้างหน้า
    y_robot = -x_cam     # ระยะซ้ายของหุ่นยนต์ = ตรงข้ามกับขวาของกล้อง
    z_robot = y_cam     # ความสูง (ขึ้น) ของหุ่นยนต์ = ตรงข้ามกับลงของกล้อง

    # ตำแหน่งของกล้องใน robot base frame (ต้องสอบเทียบจริง)
    # ตัวอย่าง: หากกล้องติดตั้งอยู่ที่ตำแหน่ง (0.2หน้า, 0.0ซ้ายชวา, 0.1บนล่าง) ใน robot base frame
    translation = np.array([0.165, 0.03, 0.125])
    rotation = np.eye(3)  # หากกล้องไม่มีการหมุนเพิ่มเติมใน robot frame

    pos_cam = np.array([x_robot, y_robot, z_robot])
    pos_robot = rotation.dot(pos_cam) + translation
    return pos_robot

# --- โหลดโมเดล YOLO ---
model = YOLO('/home/banhbucks/Mangkud_rev.pt')

# --- กำหนดค่า pipeline ของ RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

class YoloDetectionNode:
    def __init__(self):
        rospy.init_node('yolo_detection_node', anonymous=True)
        self.image_pub = rospy.Publisher('/yolo_image', Image, queue_size=10)
        self.pose_pub = rospy.Publisher('/detected_object_pose', PoseStamped, queue_size=10)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.detection_counts = defaultdict(int)
        self.detection_history = defaultdict(list)

    def run(self):
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            filtered_depth_frame = spatial.process(depth_frame)
            filtered_depth_frame = temporal.process(filtered_depth_frame)
            depth_image = np.asanyarray(filtered_depth_frame.get_data()) * depth_scale

            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            results = model(color_image, verbose=False)
            current_detections = []

            # --- ตรวจจับวัตถุและคัดกรอง bounding box ---
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    label = model.names[class_id]

                    if confidence < 0.3:
                        continue

                    depth_region = depth_image[y1:y2, x1:x2]
                    valid_depths = depth_region[depth_region > 0]
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    object_depth = np.median(valid_depths) if len(valid_depths) > 0 else 0

                    if object_depth <= 0.75:
                        current_detections.append((object_depth, x1, y1, x2, y2, class_id, center_x, center_y))

            # --- อัปเดตการตรวจจับ ---
            temp_counts = defaultdict(int)
            temp_history = defaultdict(list)

            for detection in current_detections:
                object_depth, x1, y1, x2, y2, class_id, center_x, center_y = detection
                key = (class_id, center_x // 10, center_y // 10)
                if key in self.detection_counts:
                    self.detection_counts[key] += 1
                    self.detection_history[key].append(detection)
                else:
                    self.detection_counts[key] = 1
                    self.detection_history[key] = [detection]

                temp_counts[key] = self.detection_counts[key]
                temp_history[key] = self.detection_history[key]

            self.detection_counts = temp_counts
            self.detection_history = temp_history

            confirmed_detections = []
            for key in self.detection_counts:
                if self.detection_counts[key] >= 5:
                    confirmed_detections.append(self.detection_history[key][-1])

            confirmed_detections.sort(key=lambda x: x[0])

            # --- หากมีการตรวจจับที่ยืนยันแล้ว ---
            if confirmed_detections:
                object_depth, x1, y1, x2, y2, class_id, center_x, center_y = confirmed_detections[0]
                if object_depth > 0:
                    # ใช้ RealSense SDK เพื่อคำนวณพิกัด 3D
                    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [center_x, center_y], object_depth)
                    x_cam, y_cam, z_cam = point

                    # แปลงไปยัง robot base frame
                    pos_robot = transform_to_robot_frame(x_cam, y_cam, z_cam)

                    # สร้าง PoseStamped message
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = "camera_link"
                    pose_msg.pose.position.x = pos_robot[0]
                    pose_msg.pose.position.y = pos_robot[1]
                    pose_msg.pose.position.z = pos_robot[2]

                    # กำหนด orientation (gripper เฉียงขึ้น 60 องศาจากทิศหันไปข้างหน้า)
                    q = quaternion_from_euler(0, 0, 0)  # หมุน -30 องศารอบแกน Y
                    pose_msg.pose.orientation.x = q[0]
                    pose_msg.pose.orientation.y = q[1]
                    pose_msg.pose.orientation.z = q[2]
                    pose_msg.pose.orientation.w = q[3]

                    self.pose_pub.publish(pose_msg)
                    rospy.loginfo(f"[{timestamp_str}] Published detected object pose: {pose_msg.pose.position}")

                    # วาด bounding box และข้อความ
                    text = f"{model.names[class_id]}: {object_depth*100:.2f}cm"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (252, 119, 30), 2)
                    cv2.putText(color_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

            image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            self.image_pub.publish(image_msg)
            self.rate.sleep()

    def shutdown(self):
        pipeline.stop()
        rospy.loginfo("RealSense pipeline stopped")

if __name__ == '__main__':
    node = YoloDetectionNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()
