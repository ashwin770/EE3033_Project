#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_mapper.py — Autonomous SLAM Mapping Node

Runs alongside gmapping to automatically drive the robot around the
environment and build the map, with NO manual driving required.

How it works:
  1. Reads the growing /map from gmapping.
  2. Gets the robot pose from the TF transform (map -> base_link),
     which gmapping publishes directly — no AMCL needed.
  3. Finds frontier cells (free space adjacent to unknown space).
  4. Drives to the closest frontier using move_base.
  5. When no more frontiers exist (map is complete), saves the map
     automatically and shuts down.

Usage: launched automatically by mapping.launch
"""

import rospy
import actionlib
import math
import subprocess
import os

from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Bool

import tf


class AutoMapper:
    def __init__(self):
        rospy.init_node('auto_mapper')

        # ── Parameters ──────────────────────────────────────────────────────
        # How many seconds before giving up on a stuck goal and trying the next frontier
        self.goal_timeout = rospy.get_param('~goal_timeout', 25.0)

        # Minimum distance to consider a frontier (avoids micro-steps)
        self.min_frontier_dist = rospy.get_param('~min_frontier_dist', 0.4)

        # Where to save the map (no extension — map_saver adds .yaml / .pgm)
        default_map_path = os.path.expanduser(
            '~/catkin_ws/src/ee3033_project/maps/sandbox_map'
        )
        self.map_save_path = rospy.get_param('~map_save_path', default_map_path)

        # ── State ────────────────────────────────────────────────────────────
        self.map_data = []
        self.map_info = None
        self.goal_start_time = None
        self.current_goal = None
        self.blacklist = []
        self.blacklist_radius = 0.2

        # ── TF listener for map -> base_link pose (provided by gmapping) ────
        self.tf_listener = tf.TransformListener()

        # ── ROS interfaces ───────────────────────────────────────────────────
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.complete_pub = rospy.Publisher(
            '/mapping_complete', Bool, queue_size=1, latch=True
        )

        # ── move_base client ─────────────────────────────────────────────────
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('[AutoMapper] Waiting for move_base...')
        self.client.wait_for_server()
        rospy.loginfo('[AutoMapper] Connected to move_base.')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def map_callback(self, msg):
        self.map_data = list(msg.data)
        self.map_info = msg.info

    # ── Pose from TF (gmapping publishes map -> base_link directly) ──────────

    def get_robot_pose(self):
        """Return (x, y) of the robot in the map frame, or None if not yet available."""
        try:
            self.tf_listener.waitForTransform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0)
            )
            (trans, _rot) = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0)
            )
            return trans[0], trans[1]
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return None

    # ── Frontier detection ───────────────────────────────────────────────────

    def get_frontiers(self):
        """Return world-coordinate (wx, wy) positions of frontier cells."""
        if not self.map_data or self.map_info is None:
            return []

        w = self.map_info.width
        h = self.map_info.height
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        frontiers = []
        for y in range(1, h - 1, 2):
            for x in range(1, w - 1, 2):
                idx = x + y * w
                if self.map_data[idx] == 0:          # free cell
                    for n in [idx - 1, idx + 1, idx - w, idx + w]:
                        if 0 <= n < len(self.map_data) and self.map_data[n] == -1:
                            frontiers.append((x * res + ox, y * res + oy))
                            break
        return frontiers

    def pick_closest_frontier(self, frontiers, rx, ry):
        """Return the frontier closest to (rx, ry) that is beyond min_frontier_dist."""
        best, best_dist = None, float('inf')
        for (wx, wy) in frontiers:
            is_blacklisted = any(math.sqrt((wx - bx)**2 + (wy - by)**2) < self.blacklist_radius for bx, by in self.blacklist)
            if is_blacklisted:
                continue

            d = math.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)
            if self.min_frontier_dist < d < best_dist:
                best_dist = d
                best = (wx, wy)
        return best

    # ── Navigation ───────────────────────────────────────────────────────────

    def send_goal(self, x, y):
        self.current_goal = (x, y)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0
        self.client.send_goal(goal)
        self.goal_start_time = rospy.Time.now()
        rospy.loginfo('[AutoMapper] Goal → x={:.2f}, y={:.2f}'.format(x, y))

    # ── Map saving ───────────────────────────────────────────────────────────

    def save_map(self, suffix=""):
        save_path = self.map_save_path + suffix
        rospy.loginfo('[AutoMapper] Saving map to: ' + save_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        result = subprocess.call(
            ['rosrun', 'map_server', 'map_saver', '-f', save_path]
        )
        if result == 0:
            rospy.loginfo('[AutoMapper] Map saved successfully!')
        else:
            rospy.logwarn('[AutoMapper] map_saver returned non-zero exit code.')

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(0.5)   # check every 2 seconds

        # Wait for the map and TF to become available
        rospy.loginfo('[AutoMapper] Waiting for map and TF (map->base_link)...')
        while not rospy.is_shutdown():
            if self.map_data and self.get_robot_pose() is not None:
                break
            rospy.sleep(1.0)
            
        rospy.loginfo('[AutoMapper] Ready — waiting a moment for initial laser scans...')
        rospy.sleep(3.0)
        self.save_map(suffix="_initial")
        rospy.loginfo('[AutoMapper] Saved one-shot initial map. Starting autonomous mapping!')
        
        last_periodic_save = rospy.Time.now()

        while not rospy.is_shutdown():
            state = self.client.get_state()
            now = rospy.Time.now()
            goal_active = state in [GoalStatus.ACTIVE, GoalStatus.PENDING]
            
            # Periodic saving every 20 seconds
            if (now - last_periodic_save).to_sec() >= 20.0:
                self.save_map()
                last_periodic_save = now

            # Check for aborted/rejected goals to blacklist
            if not goal_active and self.current_goal is not None:
                if state in [GoalStatus.ABORTED, GoalStatus.REJECTED]:
                    rospy.logwarn('[AutoMapper] Goal aborted by move_base. Blacklisting frontier.')
                    self.blacklist.append(self.current_goal)
                self.current_goal = None

            # Cancel if stuck
            if goal_active and self.goal_start_time is not None:
                elapsed = (now - self.goal_start_time).to_sec()
                if elapsed > self.goal_timeout:
                    rospy.logwarn(
                        '[AutoMapper] Goal timed out after {:.0f}s — skipping & blacklisting.'.format(elapsed)
                    )
                    self.client.cancel_goal()
                    if self.current_goal:
                        self.blacklist.append(self.current_goal)
                        self.current_goal = None
                    goal_active = False

            if not goal_active:
                pose = self.get_robot_pose()
                if pose is None:
                    rospy.logwarn('[AutoMapper] Cannot get robot pose yet.')
                    rate.sleep()
                    continue

                rx, ry = pose
                frontiers = self.get_frontiers()

                if not frontiers:
                    rospy.loginfo('[AutoMapper] No frontiers left — map is complete!')
                    self.client.cancel_all_goals()
                    self.save_map()
                    self.complete_pub.publish(Bool(data=True))
                    break

                target = self.pick_closest_frontier(frontiers, rx, ry)
                if target is None:
                    rospy.loginfo('[AutoMapper] All frontiers too close — waiting...')
                else:
                    self.send_goal(target[0], target[1])

            rate.sleep()

        rospy.loginfo('[AutoMapper] Autonomous mapping finished.')


if __name__ == '__main__':
    try:
        mapper = AutoMapper()
        mapper.run()
    except rospy.ROSInterruptException:
        pass
