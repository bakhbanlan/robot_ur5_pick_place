#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler
import math 
from ur_msgs.srv import SetIO, SetIORequest

class UR5eController:
    def __init__(self):
        # เริ่มต้น moveit และ ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur5e_pick_place', anonymous=True)
        
        # ตั้งค่า MoveGroupCommander สำหรับกลุ่ม 'manipulator'
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        self.move_group.set_planning_time(20)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(0.5)
        self.move_group.set_max_acceleration_scaling_factor(0.5)
        
        # สร้าง service proxy สำหรับควบคุม gripper
        rospy.wait_for_service('/ur_hardware_interface/set_io', timeout=10)
        self.io_client = rospy.ServiceProxy('/ur_hardware_interface/set_io', SetIO)
        
        rospy.sleep(2)
        rospy.loginfo("Current joint values: %s", self.move_group.get_current_joint_values())
        rospy.loginfo("Current pose: %s", self.move_group.get_current_pose().pose)
        
        # Subscriber สำหรับรับตำแหน่งของวัตถุ (PoseStamped)
        self.pose_sub = rospy.Subscriber('/detected_object_pose', PoseStamped, self.pose_callback)
        self.object_pose = None
        self.received_pose = False

        # กำหนดตำแหน่งที่ใช้ในระบบ
        self.start_joint_angles = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        self.place_joint_angles = [-1.2, -1.5, 1.5, -1.57, -1.5, 0.0]
        
    
    def pose_callback(self, msg):
        rospy.loginfo("Received detected object pose")
        self.object_pose = msg.pose
        self.received_pose = True

    def move_ur5e_to(self, joint_angles):
        self.move_group.set_joint_value_target(joint_angles)
        rospy.loginfo("Moving UR5e to joint angles: %s", joint_angles)
        
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if not success:
            rospy.logerr("Motion execution failed!")
            return False
        
        rospy.loginfo("Motion executed successfully.")
        return True
    
    def reset_gripper(self):
        """เคลียร์ค่า Tool Digital Output 0 และ 1"""
        self.set_io(16, 0)  # Tool Digital Output 0 = 0
        self.set_io(17, 0)  # Tool Digital Output 1 = 0
        rospy.loginfo('Gripper: Resetting Tool Digital Outputs')

    def set_io(self, pin, state):
        """ควบคุมค่า digital I/O ด้วย service call"""
        req = SetIORequest()
        req.fun = 1  # 1 = Digital Output
        req.pin = pin  # ระบุ pin ที่ต้องการควบคุม
        req.state = state  # ค่า state (0.0 = ปิด, 1.0 = เปิด)
        try:
            self.io_client(req)
            rospy.loginfo(f"Set pin {pin} to state {state}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to set IO: {e}")

    def open_gripper(self):
        """เปิด gripper โดยใช้ service call"""
        self.set_io(16, 1.0)  # เปิด (Pin 16 = 1)
        self.set_io(17, 0.0)  # ปิด (Pin 17 = 0)
        rospy.loginfo("Gripper Opened")
        rospy.sleep(2)  # รอให้ gripper เปิดเสร็จก่อนทำงานต่อ

    def close_gripper(self):
        """ปิด gripper โดยใช้ service call"""
        self.set_io(16, 0.0)  # ปิด (Pin 16 = 0)
        self.set_io(17, 1.0)  # เปิด (Pin 17 = 1)
        rospy.loginfo("Gripper Closed")
        rospy.sleep(2)  # รอให้ gripper ปิดเสร็จก่อนทำงานต่อ
        
    def generate_pick_pose(self, x, y, z):
        """สร้างตำแหน่ง pick pose ด้วยพิกัด x, y, z และกำหนด orientation ให้ gripper หันลง"""
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        # กำหนด orientation ให้ gripper หันลง (rotation: -pi รอบ X, 0 รอบ Y, pi/2 รอบ Z)
        q = quaternion_from_euler(-math.pi, 0, math.pi/2)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        rospy.loginfo("Target Pick Pose: position(%.2f, %.2f, %.2f) orientation(%f, %f, %f, %f)",
                      x, y, z, q[0], q[1], q[2], q[3])
        return target_pose

    def move_with_cartesian_path(self, waypoints):
        rospy.loginfo("Executing Cartesian path...")

        if not isinstance(waypoints, list) or not all(isinstance(wp, Pose) for wp in waypoints):
            rospy.logerr("Invalid waypoints format! Must be a list of Pose objects.")
            return False

        eef_step = 0.01  # ระยะก้าวเล็ก ๆ
        jump_threshold_bool = False  # ส่งเป็น bool (False)

        # เรียกใช้ compute_cartesian_path ด้วย 3 พารามิเตอร์: waypoints, eef_step, jump_threshold_bool
        plan, fraction = self.move_group.compute_cartesian_path(waypoints, eef_step, jump_threshold_bool)
        
        if fraction < 0.9:
            rospy.logwarn("Only {:.2f}% of the Cartesian path was planned!".format(fraction * 100))
            return False

        rospy.loginfo("Executing Cartesian path with {:.2f}% success.".format(fraction * 100))
        return self.move_group.execute(plan, wait=True)

    def pick_and_place(self, pick_position):
        if pick_position is None:
            rospy.logerr("Pick position is None! Skipping pick and place.")
            return False

        # เปิด gripper ก่อนเริ่มเคลื่อนที่ไปตำแหน่ง pick
        self.open_gripper()
        self.reset_gripper()

        # เคลื่อนที่ไปยังตำแหน่ง pick ด้วย Cartesian path
        rospy.loginfo("Moving to pick position")
        pick_pose = self.generate_pick_pose(*pick_position)
        waypoints = [self.move_group.get_current_pose().pose, pick_pose]
        if not self.move_with_cartesian_path(waypoints):
            rospy.logerr("Failed to move to pick position.")
            return False
        
        # เมื่อถึงตำแหน่ง pick ให้ปิด gripper
        rospy.loginfo("At pick position, closing gripper")
        self.close_gripper()
        self.reset_gripper()
        
        # เคลื่อนที่ไปยังตำแหน่ง place แล้วเปิด gripper
        rospy.loginfo("Moving to place position")
        if not self.move_ur5e_to(self.place_joint_angles):
            rospy.logerr("Failed to move to place position.")
            return False
        rospy.loginfo("At place position, opening gripper")
        self.open_gripper()
        self.reset_gripper()

        # เคลื่อนที่ไปยังตำแหน่ง end 
        rospy.loginfo("Moving to end position")
        if not self.move_ur5e_to(self.start_joint_angles):
            rospy.logerr("Failed to move to end position.")
            return False

        return True

    def run(self):
        rospy.loginfo("UR5e Controller running, starting loop cycle...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # 1. เคลื่อนที่ไปยัง start position
            rospy.loginfo("Moving to start position")
            if not self.move_ur5e_to(self.start_joint_angles):
                rospy.logerr("Failed to move to start position, retrying...")
                continue
            self.open_gripper()
            self.reset_gripper()

            # 2. รอรับตำแหน่ง pick position จาก subscriber
            rospy.loginfo("Waiting for pick position...")
            while not self.received_pose and not rospy.is_shutdown():
                rate.sleep()
            if rospy.is_shutdown():
                break
            pick_position = [
                self.object_pose.position.x,
                self.object_pose.position.y,
                self.object_pose.position.z,
            ]
            rospy.loginfo("Pick position received: %s", pick_position)

            # เรียกใช้ pick_and_place sequence ตามขั้นตอนที่กำหนด
            if not self.pick_and_place(pick_position):
                rospy.logerr("Pick and place sequence failed, resetting cycle.")
                self.received_pose = False
                self.object_pose = None
                continue

            # รีเซ็ตสถานะสำหรับรอบถัดไป แล้ววนกลับไปที่ start
            self.received_pose = False
            self.object_pose = None
            rospy.loginfo("Cycle completed, restarting...")
            rospy.sleep(1)

    def shutdown(self):
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    controller = UR5eController()   
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.shutdown()