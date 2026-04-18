#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import threading

import rospy
import actionlib
import tf

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, PointStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes


def is_finite_number(value):
    return (not math.isinf(value)) and (not math.isnan(value))


class YoloObjectSearch(object):
    def __init__(self):
        rospy.init_node("yolo_object_search_ros1")

        # Frames / target
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.target_class = rospy.get_param("~target_class", "bottle").strip().lower()

        # Image size
        self.image_width = int(rospy.get_param("~image_width", 640))
        self.image_height = int(rospy.get_param("~image_height", 480))

        # Search behavior
        self.scan_sector_deg = float(rospy.get_param("~scan_sector_deg", 25.0))
        self.scan_hold_time = float(rospy.get_param("~scan_hold_time", 15.0))
        self.scan_turn_speed = float(rospy.get_param("~scan_turn_speed", 0.18))
        self.max_full_cycles = int(rospy.get_param("~max_full_cycles", 1000))

        # Nav goal behavior
        self.move_timeout = float(rospy.get_param("~move_timeout", 40.0))
        self.return_timeout = float(rospy.get_param("~return_timeout", 120.0))
        self.nav_goal_wait_timeout = float(rospy.get_param("~nav_goal_wait_timeout", 0.0))  # 0 means wait forever

        # Detection thresholds
        self.search_confidence_min = float(rospy.get_param("~search_confidence_min", 0.25))
        self.approach_confidence_min = float(rospy.get_param("~approach_confidence_min", 0.25))
        self.detection_timeout = float(rospy.get_param("~detection_timeout", 1.0))

        # Alignment / approach
        self.centering_kp = float(rospy.get_param("~centering_kp", 0.0035))
        self.max_turn_speed = float(rospy.get_param("~max_turn_speed", 0.28))
        self.align_tolerance_ratio = float(rospy.get_param("~align_tolerance_ratio", 0.18))
        self.align_stable_cycles = int(rospy.get_param("~align_stable_cycles", 2))
        self.align_timeout = float(rospy.get_param("~align_timeout", 10.0))

        self.forward_speed = float(rospy.get_param("~forward_speed", 0.10))
        self.medium_forward_speed = float(rospy.get_param("~medium_forward_speed", 0.06))
        self.slow_forward_speed = float(rospy.get_param("~slow_forward_speed", 0.03))
        self.approach_center_tolerance_ratio = float(rospy.get_param("~approach_center_tolerance_ratio", 0.22))
        self.forward_motion_center_tolerance_ratio = float(rospy.get_param("~forward_motion_center_tolerance_ratio", 0.40))

        # Visual stop logic
        self.visual_return_width_ratio = float(rospy.get_param("~visual_return_width_ratio", 0.28))
        self.visual_return_bottom_ratio = float(rospy.get_param("~visual_return_bottom_ratio", 0.82))
        self.very_close_width_ratio = float(rospy.get_param("~very_close_width_ratio", 0.55))
        self.object_lost_retry_limit = int(rospy.get_param("~object_lost_retry_limit", 8))
        self.min_approach_distance = float(rospy.get_param("~min_approach_distance", 0.05))

        # LiDAR safety only
        self.front_angle_deg = float(rospy.get_param("~front_angle_deg", 8.0))
        self.stop_distance = float(rospy.get_param("~stop_distance", 0.20))
        self.max_valid_lidar_distance = float(rospy.get_param("~max_valid_lidar_distance", 3.0))
        self.min_valid_lidar_distance = float(rospy.get_param("~min_valid_lidar_distance", 0.08))
        self.scan_stale_timeout = float(rospy.get_param("~scan_stale_timeout", 1.0))

        # Runtime state
        self.lock = threading.Lock()
        self.home_pose = None
        self.home_received = False
        self.current_pose = None

        self.latest_target_box = None
        self.latest_detection_time = rospy.Time(0)

        self.latest_scan = None
        self.latest_scan_time = rospy.Time(0)

        self.pending_nav_goal = None
        self.pending_nav_goal_seq = 0
        self.last_consumed_nav_goal_seq = 0

        self.search_cycle_index = 0
        self.object_lost_count = 0
        self.mission_done = False
        self.is_shutting_down = False

        # Publishers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.status_pub = rospy.Publisher("/search_status", String, queue_size=10)
        self.object_location_pub = rospy.Publisher("/found_object_location", PointStamped, queue_size=1, latch=True)
        self.robot_stop_pose_pub = rospy.Publisher("/robot_stop_pose", PoseStamped, queue_size=1, latch=True)
        self.object_text_pub = rospy.Publisher("/found_object_text", String, queue_size=1, latch=True)

        # Subscribers
        self.initialpose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initialpose_cb, queue_size=1)
        self.amcl_pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_pose_cb, queue_size=1)
        self.bounding_boxes_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bounding_boxes_cb, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_cb, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.nav_goal_cb, queue_size=1)

        rospy.on_shutdown(self.cleanup)

        self.tf_listener = tf.TransformListener()
        self.move_base_client = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        rospy.loginfo("Waiting for /move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to /move_base.")

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    def cleanup(self):
        if self.is_shutting_down:
            return

        self.is_shutting_down = True

        try:
            self.stop_robot()
        except Exception:
            pass

        try:
            self.cancel_nav()
        except Exception:
            pass

        subs = [
            getattr(self, "initialpose_sub", None),
            getattr(self, "amcl_pose_sub", None),
            getattr(self, "bounding_boxes_sub", None),
            getattr(self, "scan_sub", None),
            getattr(self, "goal_sub", None),
        ]

        for sub in subs:
            if sub is not None:
                try:
                    sub.unregister()
                except Exception:
                    pass

        try:
            rospy.sleep(0.2)
        except Exception:
            pass

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def initialpose_cb(self, msg):
        if self.is_shutting_down:
            return

        with self.lock:
            self.home_pose = PoseStamped()
            self.home_pose.header.frame_id = self.map_frame
            self.home_pose.header.stamp = rospy.Time.now()
            self.home_pose.pose = msg.pose.pose
            self.home_received = True

        p = msg.pose.pose.position
        yaw = self.yaw_from_quaternion(msg.pose.pose.orientation)
        rospy.loginfo("Initial pose saved: x=%.3f y=%.3f yaw=%.2f deg", p.x, p.y, math.degrees(yaw))

    def amcl_pose_cb(self, msg):
        if self.is_shutting_down:
            return
        with self.lock:
            self.current_pose = msg.pose.pose

    def scan_cb(self, msg):
        if self.is_shutting_down:
            return
        with self.lock:
            self.latest_scan = msg
            self.latest_scan_time = rospy.Time.now()

    def nav_goal_cb(self, msg):
        if self.is_shutting_down:
            return

        with self.lock:
            self.pending_nav_goal = msg
            self.pending_nav_goal_seq += 1

        rospy.loginfo("Received RViz 2D Nav Goal: x=%.3f y=%.3f",
                      msg.pose.position.x, msg.pose.position.y)

    def bounding_boxes_cb(self, msg):
        if self.is_shutting_down:
            return

        best_box = None
        best_prob = -1.0
        for box in msg.bounding_boxes:
            cls = box.Class.strip().lower()
            prob = float(box.probability)
            if cls == self.target_class and prob >= self.search_confidence_min:
                if prob > best_prob:
                    best_prob = prob
                    best_box = box

        with self.lock:
            if best_box is not None:
                self.latest_target_box = best_box
                self.latest_detection_time = rospy.Time.now()

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def publish_status(self, text):
        rospy.loginfo(text)
        self.status_pub.publish(String(data=text))

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def quaternion_from_yaw(self, yaw):
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        quat = Quaternion()
        quat.x, quat.y, quat.z, quat.w = q[0], q[1], q[2], q[3]
        return quat

    def yaw_from_quaternion(self, q):
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(quat)
        return yaw

    def normalize_angle(self, ang):
        while ang > math.pi:
            ang -= 2.0 * math.pi
        while ang < -math.pi:
            ang += 2.0 * math.pi
        return ang

    def angle_diff(self, a, b):
        return self.normalize_angle(a - b)

    def distance_xy(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def wait_for_initial_pose(self):
        self.publish_status("Waiting for RViz 2D Pose Estimate on /initialpose...")
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            with self.lock:
                if self.home_received:
                    return True
            rate.sleep()
        return False

    def wait_for_user_nav_goal(self):
        self.publish_status("Target not found after full 360 scan. Please click a 2D Nav Goal in RViz.")

        rate = rospy.Rate(5)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.mission_done:
                return None

            with self.lock:
                seq = self.pending_nav_goal_seq
                goal = self.pending_nav_goal

            if seq > self.last_consumed_nav_goal_seq and goal is not None:
                self.last_consumed_nav_goal_seq = seq
                return goal

            if self.nav_goal_wait_timeout > 0.0:
                if (rospy.Time.now() - start_time).to_sec() > self.nav_goal_wait_timeout:
                    self.publish_status("Timed out waiting for 2D Nav Goal.")
                    return None

            rate.sleep()

        return None

    def current_box(self, require_fresh=True, min_prob=None):
        if min_prob is None:
            min_prob = self.approach_confidence_min

        with self.lock:
            box = self.latest_target_box
            t = self.latest_detection_time

        if box is None:
            return None
        if float(box.probability) < min_prob:
            return None
        if require_fresh and (rospy.Time.now() - t).to_sec() > self.detection_timeout:
            return None
        return box

    def get_robot_pose_map(self):
        with self.lock:
            pose = self.current_pose
        if pose is not None:
            return pose

        try:
            self.tf_listener.waitForTransform(self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame, self.base_frame, rospy.Time(0))
            pose = PoseStamped().pose
            pose.position.x = trans[0]
            pose.position.y = trans[1]
            pose.position.z = trans[2]
            pose.orientation.x = rot[0]
            pose.orientation.y = rot[1]
            pose.orientation.z = rot[2]
            pose.orientation.w = rot[3]
            return pose
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "TF pose lookup failed: %s", str(exc))
            return None

    def front_distance(self):
        with self.lock:
            scan = self.latest_scan
            scan_time = self.latest_scan_time

        if scan is None:
            return None
        if (rospy.Time.now() - scan_time).to_sec() > self.scan_stale_timeout:
            return None

        half = math.radians(self.front_angle_deg)
        values = []
        a = scan.angle_min
        for r in scan.ranges:
            if abs(a) <= half and is_finite_number(r):
                if self.min_valid_lidar_distance <= r <= self.max_valid_lidar_distance:
                    values.append(r)
            a += scan.angle_increment

        if not values:
            return None
        return min(values)

    def obstacle_too_close(self):
        d = self.front_distance()
        if d is None:
            return False
        return d <= self.stop_distance

    def box_center_error_ratio(self, box):
        if self.image_width <= 0:
            return None
        cx = 0.5 * (float(box.xmin) + float(box.xmax))
        return (cx - 0.5 * float(self.image_width)) / float(self.image_width)

    def box_width_ratio(self, box):
        if self.image_width <= 0:
            return 0.0
        return max(0.0, float(box.xmax - box.xmin) / float(self.image_width))

    def visual_stop_reached(self, box, width_thresh, bottom_thresh):
        if box is None:
            return False

        err_ratio = self.box_center_error_ratio(box)
        if err_ratio is None:
            return False

        width_ratio = self.box_width_ratio(box)
        bottom_ratio = float(box.ymax) / float(self.image_height)

        centered = abs(err_ratio) < self.approach_center_tolerance_ratio
        return centered and (width_ratio >= width_thresh or bottom_ratio >= bottom_thresh)

    # ------------------------------------------------------------
    # Immediate close-target return-home check
    # ------------------------------------------------------------
    def check_close_target_and_return_home(self):
        if self.mission_done or self.is_shutting_down:
            return True

        box = self.current_box(require_fresh=True, min_prob=self.search_confidence_min)
        if box is None:
            return False

        if not self.visual_stop_reached(box, self.visual_return_width_ratio, self.visual_return_bottom_ratio):
            return False

        self.stop_robot()
        rospy.sleep(0.2)
        self.stop_robot()

        self.publish_status("Target '%s' is visually close. Publishing pose and returning home." %
                            self.target_class)

        published = self.publish_found_object_result()
        if not published:
            self.publish_status("Pose publication incomplete, returning home anyway.")

        returned = self.return_home()
        if returned:
            self.publish_status("Mission complete.")
        else:
            self.publish_status("Return home failed. Mission ended at current pose.")

        self.mission_done = True
        self.cleanup()
        return True

    def rotate_in_place(self, target_delta_deg, angular_speed):
        target_delta = math.radians(target_delta_deg)
        start_pose = self.get_robot_pose_map()
        if start_pose is None:
            return False
        start_yaw = self.yaw_from_quaternion(start_pose.orientation)

        cmd = Twist()
        cmd.angular.z = abs(angular_speed) if target_delta >= 0.0 else -abs(angular_speed)
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.check_close_target_and_return_home():
                return True

            pose = self.get_robot_pose_map()
            if pose is None:
                self.stop_robot()
                return False

            yaw = self.yaw_from_quaternion(pose.orientation)
            diff = self.angle_diff(yaw, start_yaw)
            if abs(diff) >= abs(target_delta):
                break

            if self.current_box(require_fresh=True, min_prob=self.search_confidence_min) is not None:
                self.stop_robot()
                return True

            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        self.stop_robot()
        return True

    def move_base_goal(self, x, y, yaw=None, timeout=None):
        if timeout is None:
            timeout = self.move_timeout

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        if yaw is None:
            pose = self.get_robot_pose_map()
            yaw = 0.0 if pose is None else self.yaw_from_quaternion(pose.orientation)
        goal.target_pose.pose.orientation = self.quaternion_from_yaw(yaw)

        self.move_base_client.send_goal(goal)
        finished = self.move_base_client.wait_for_result(rospy.Duration(timeout))
        if not finished:
            self.move_base_client.cancel_goal()
            return False

        state = self.move_base_client.get_state()
        return state == GoalStatus.SUCCEEDED

    def move_to_user_goal(self, goal_msg):
        if goal_msg is None:
            return False

        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y
        yaw = self.yaw_from_quaternion(goal_msg.pose.orientation)

        self.publish_status("Moving to user-provided 2D Nav Goal: x=%.2f y=%.2f" % (gx, gy))
        return self.move_base_goal(gx, gy, yaw=yaw, timeout=self.move_timeout)

    def cancel_nav(self):
        self.move_base_client.cancel_all_goals()

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------
    def scan_one_sector(self, sector_idx):
        self.publish_status("Scanning sector %d ..." % sector_idx)
        rate = rospy.Rate(15)
        t0 = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.check_close_target_and_return_home():
                return True

            if self.current_box(require_fresh=True, min_prob=self.search_confidence_min) is not None:
                self.stop_robot()
                return True

            if (rospy.Time.now() - t0).to_sec() >= self.scan_hold_time:
                self.stop_robot()
                return False

            rate.sleep()

        return False

    def do_full_360_search(self):
        sectors = int(round(360.0 / self.scan_sector_deg))
        for i in range(sectors):
            found = self.scan_one_sector(i + 1)
            if self.mission_done:
                return True
            if found:
                return True

            if i != sectors - 1:
                self.rotate_in_place(self.scan_sector_deg, self.scan_turn_speed)
                if self.mission_done:
                    return True
                if self.current_box(require_fresh=True, min_prob=self.search_confidence_min) is not None:
                    return True
        return False

    # ------------------------------------------------------------
    # Approach / localization
    # ------------------------------------------------------------
    def align_to_object(self):
        self.publish_status("Object detected. Aligning to target...")
        stable = 0
        lost_count = 0
        rate = rospy.Rate(20)
        t0 = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.check_close_target_and_return_home():
                return True

            box = self.current_box(require_fresh=True, min_prob=self.approach_confidence_min)

            if box is None:
                lost_count += 1
                if lost_count >= self.object_lost_retry_limit:
                    self.stop_robot()
                    return False
                cmd = Twist()
                cmd.angular.z = 0.10
                self.cmd_vel_pub.publish(cmd)
                rate.sleep()
                continue

            lost_count = 0
            err_ratio = self.box_center_error_ratio(box)
            width_ratio = self.box_width_ratio(box)

            if err_ratio is None:
                self.stop_robot()
                return False

            if width_ratio >= self.very_close_width_ratio:
                self.stop_robot()
                return True

            if abs(err_ratio) < self.align_tolerance_ratio:
                stable += 1
            else:
                stable = 0

            if stable >= self.align_stable_cycles:
                self.stop_robot()
                return True

            if (rospy.Time.now() - t0).to_sec() > self.align_timeout:
                if abs(err_ratio) < 0.25:
                    self.stop_robot()
                    return True
                self.stop_robot()
                return False

            cmd = Twist()
            cmd.angular.z = max(-self.max_turn_speed,
                                min(self.max_turn_speed, -self.centering_kp * err_ratio * 1000.0))
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        self.stop_robot()
        return False

    def estimate_object_point_map(self):
        pose = self.get_robot_pose_map()
        if pose is None:
            return None

        yaw = self.yaw_from_quaternion(pose.orientation)
        front_d = self.front_distance()
        if front_d is None:
            front_d = 0.30

        est_d = min(front_d, self.max_valid_lidar_distance)

        pt = PointStamped()
        pt.header.frame_id = self.map_frame
        pt.header.stamp = rospy.Time.now()
        pt.point.x = pose.position.x + est_d * math.cos(yaw)
        pt.point.y = pose.position.y + est_d * math.sin(yaw)
        pt.point.z = 0.0
        return pt

    def publish_robot_stop_pose(self):
        pose = self.get_robot_pose_map()
        if pose is None:
            self.publish_status("Could not read robot stop pose.")
            return False

        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.Time.now()
        msg.pose = pose
        self.robot_stop_pose_pub.publish(msg)

        yaw = self.yaw_from_quaternion(pose.orientation)
        self.publish_status("Robot stop pose: x=%.3f, y=%.3f, z=%.3f, yaw=%.3f rad" %
                            (pose.position.x, pose.position.y, pose.position.z, yaw))
        return True

    def approach_object(self):
        self.publish_status("Approaching object while keeping alignment...")
        self.object_lost_count = 0
        rate = rospy.Rate(20)

        start_pose = self.get_robot_pose_map()
        start_x = None
        start_y = None
        if start_pose is not None:
            start_x = start_pose.position.x
            start_y = start_pose.position.y

        while not rospy.is_shutdown():
            if self.check_close_target_and_return_home():
                return True

            box = self.current_box(require_fresh=True, min_prob=self.approach_confidence_min)

            if box is None:
                self.object_lost_count += 1
                self.stop_robot()
                if self.object_lost_count >= self.object_lost_retry_limit:
                    self.publish_status("Target lost during approach.")
                    return False
                rate.sleep()
                continue

            self.object_lost_count = 0
            err_ratio = self.box_center_error_ratio(box)
            width_ratio = self.box_width_ratio(box)

            if err_ratio is None:
                self.stop_robot()
                return False

            current_pose = self.get_robot_pose_map()
            traveled = 0.0
            if current_pose is not None and start_x is not None and start_y is not None:
                traveled = self.distance_xy(start_x, start_y,
                                            current_pose.position.x, current_pose.position.y)

            centered = abs(err_ratio) < self.approach_center_tolerance_ratio
            can_move_forward = abs(err_ratio) < self.forward_motion_center_tolerance_ratio
            very_close_visually = width_ratio >= self.very_close_width_ratio

            if centered and very_close_visually and traveled >= self.min_approach_distance:
                self.stop_robot()
                rospy.sleep(0.2)
                self.stop_robot()
                self.publish_status("Reached target using visual close-range confirmation.")
                return True

            if self.obstacle_too_close():
                if centered:
                    self.stop_robot()
                    rospy.sleep(0.2)
                    self.stop_robot()
                    self.publish_status("Stopped by close obstacle while centered on target.")
                    return True
                self.stop_robot()
                self.publish_status("Obstacle in front before confirmed target reach. Approach aborted.")
                return False

            cmd = Twist()
            cmd.angular.z = max(-self.max_turn_speed,
                                min(self.max_turn_speed, -self.centering_kp * err_ratio * 1000.0))

            if can_move_forward:
                front_d = self.front_distance()
                if front_d is not None:
                    if front_d <= 0.40:
                        cmd.linear.x = self.slow_forward_speed
                    elif front_d <= 0.60:
                        cmd.linear.x = self.medium_forward_speed
                    else:
                        cmd.linear.x = self.forward_speed
                else:
                    cmd.linear.x = self.slow_forward_speed
            else:
                cmd.linear.x = 0.0

            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        self.stop_robot()
        return False

    def publish_found_object_result(self):
        ok_robot = self.publish_robot_stop_pose()

        pt = self.estimate_object_point_map()
        if pt is None:
            self.publish_status("Could not estimate object map point from LiDAR.")
            return ok_robot

        text = "Detected '%s' at map coordinates x=%.3f, y=%.3f, z=%.3f" % (
            self.target_class, pt.point.x, pt.point.y, pt.point.z)
        self.object_location_pub.publish(pt)
        self.object_text_pub.publish(String(data=text))
        self.publish_status(text)
        return True

    # ------------------------------------------------------------
    # Return home
    # ------------------------------------------------------------
    def return_home(self):
        with self.lock:
            home = self.home_pose
        if home is None:
            self.publish_status("No saved home pose. Cannot return.")
            return False

        self.stop_robot()
        rospy.sleep(0.3)
        self.cancel_nav()

        x = home.pose.position.x
        y = home.pose.position.y
        current_pose = self.get_robot_pose_map()
        if current_pose is not None:
            yaw = self.yaw_from_quaternion(current_pose.orientation)
        else:
            yaw = self.yaw_from_quaternion(home.pose.orientation)

        self.publish_status("Returning to start pose x=%.2f y=%.2f" % (x, y))
        ok = self.move_base_goal(x, y, yaw=yaw, timeout=self.return_timeout)
        if ok:
            self.publish_status("Returned to initial pose.")
        else:
            self.publish_status("Return to initial pose failed.")
        return ok

    # ------------------------------------------------------------
    # Main mission
    # ------------------------------------------------------------
    def run(self):
        if not self.wait_for_initial_pose():
            return

        self.publish_status("Mission started for target class: %s" % self.target_class)

        for cycle in range(self.max_full_cycles):
            if rospy.is_shutdown():
                break

            if self.mission_done:
                return

            self.search_cycle_index = cycle + 1
            self.publish_status("=== Search cycle %d ===" % self.search_cycle_index)

            found = self.do_full_360_search()
            if self.mission_done:
                return

            if found:
                self.cancel_nav()
                self.stop_robot()

                aligned = self.align_to_object()
                if self.mission_done:
                    return

                if not aligned:
                    self.publish_status("Alignment failed. Retrying detection before next scan.")
                    rospy.sleep(0.5)
                    if self.current_box(require_fresh=True, min_prob=self.search_confidence_min) is not None:
                        aligned = self.align_to_object()

                if self.mission_done:
                    return

                if not aligned:
                    self.publish_status("Alignment failed. Resuming search.")
                    continue

                reached = self.approach_object()
                if self.mission_done:
                    return

                if not reached:
                    self.publish_status("Approach failed. Resuming search.")
                    continue

                self.stop_robot()
                rospy.sleep(0.2)
                self.stop_robot()

                published = self.publish_found_object_result()
                if not published:
                    self.publish_status("Pose publication incomplete, but returning home anyway.")

                returned = self.return_home()
                if returned:
                    self.publish_status("Mission complete.")
                else:
                    self.publish_status("Return home failed. Mission ended at current pose.")

                self.mission_done = True
                self.cleanup()
                return

            # No target found: wait for user 2D Nav Goal
            goal_msg = self.wait_for_user_nav_goal()
            if goal_msg is None:
                self.publish_status("No user 2D Nav Goal received. Staying at current pose.")
                continue

            moved = self.move_to_user_goal(goal_msg)
            if moved:
                self.publish_status("Reached user 2D Nav Goal. Starting next 360 scan.")
            else:
                self.publish_status("Failed to reach user 2D Nav Goal. Waiting for another goal.")


if __name__ == "__main__":
    node = None
    try:
        node = YoloObjectSearch()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if node is not None:
            try:
                node.cleanup()
            except Exception:
                pass
