#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from styx_msgs.msg import Lane, Waypoint

import math
import tf
import numpy as np

from enum import Enum

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

MAX_DECEL = 1.0
LOOKAHEAD_WPS = 80 # Number of waypoints we will publish. You can change this number

NO_CAMERA_YET = -2
NO_RED = -1

LOOP_RATE = 2

SLOW_START_SPEED = 2.0

class State(Enum):
    NoCameraYet = 0
    SlowStart = 1
    Going = 2 
    Stopping = 3
    EndOfSlowStart = 4        

class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.current_pose = None
        self.light_wp = NO_CAMERA_YET
        self.state = State.NoCameraYet  

        self.loop()

    def loop(self):
        rate = rospy.Rate(LOOP_RATE) # 10 # 40 # Hz

        while not rospy.is_shutdown():
            if self.current_pose and self.base_waypoints:
                next = self.get_next_waypoint()
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                #lane.waypoints = self.just_go(next)

                if self.light_wp == NO_CAMERA_YET:  # no camera
                    self.state = State.NoCameraYet
                    lane.waypoints = self.no_go()

                elif self.light_wp == NO_RED:  # free to go
                    self.state = State.Going
                    lane.waypoints = self.just_go(next)

                else:  # red light found
                    if self.state == State.NoCameraYet or self.state == State.SlowStart:  # slow starting, with camera
                        self.state = State.SlowStart   # slow-start
                        lane.waypoints = self.slow_down(next, self.light_wp, slow_start=True)
                        if len(lane.waypoints) <= 0:
                            self.state = State.EndOfSlowStart

                    elif self.state == State.EndOfSlowStart:  # end of slow start
                        self.state = State.EndOfSlowStart
                        lane.waypoints = self.no_go()

                    elif next < self.light_wp:  # going or stopping, and did not reach stop line yet
                        self.state = State.Stopping  # stopping
                        lane.waypoints = self.slow_down(next, self.light_wp)

                    elif self.state == State.Stopping:  # stopping already, doing nothing
                        self.state = State.Stopping
                        lane.waypoints = self.no_go()

                    else:  # going past red light: apparently a sudden switch right under the light, continuing going
                        self.state = State.Going
                        lane.waypoints = self.just_go(next)

                self.final_waypoints_pub.publish(lane)
                print(self.state)

            rate.sleep()

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    # ACTIONS
    def no_go(self):
        return []

    def just_go(self, next):
        return self.base_waypoints[next : next + LOOKAHEAD_WPS]

    def slow_down(self, next, down_to, slow_start=False):
        stopping_waypoints = []
  
        if next < down_to:
            for wp in self.base_waypoints[next : down_to + 1]:
                p = Waypoint()
                wp_pose = wp.pose.pose
                p.pose.pose.position.x = wp_pose.position.x
                p.pose.pose.position.y = wp_pose.position.y
                p.pose.pose.position.z = wp_pose.position.z
  
                q = self.quaternion_from_yaw(0.0)
                p.pose.pose.orientation = Quaternion(*q)
  
                p.twist.twist.linear.x = SLOW_START_SPEED if slow_start else wp.twist.twist.linear.x
                stopping_waypoints.append(p)
  
        return self.decelerate(stopping_waypoints) if len(stopping_waypoints) > 0 else []

    def distance_sq(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return x*x + y*y + z*z

    def get_closest_waypoint(self):
        self_pos = self.current_pose.pose.position
        distances = np.array([self.distance_sq(self_pos, way_pos.pose.pose.position) for way_pos in self.base_waypoints])
        return np.argmin(distances)

    # https://answers.ros.org/question/69754/quaternion-transformations-in-python/
    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def yaw_from_quaternion(self, quaternion):
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def get_next_waypoint(self):
        closest = self.get_closest_waypoint()

        orientation = self.current_pose.pose.orientation
        self_pos = self.current_pose.pose.position
        way_pos = self.base_waypoints[closest].pose.pose.position
        
        heading = math.atan2((way_pos.y-self_pos.y), (way_pos.x-self_pos.x))
        yaw = self.yaw_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        angle = abs(yaw - heading);

        if (angle > math.pi/4):
            return closest + 1

        return closest

    def pose_cb(self, msg):
        self.current_pose = msg
        pass

    def waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints;
        pass

    def traffic_cb(self, msg):
        self.light_wp = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
