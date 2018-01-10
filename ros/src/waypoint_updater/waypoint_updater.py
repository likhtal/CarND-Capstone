#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import tf
import numpy as np

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

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.base_waypoints = None

        self.current_pose = None
 
        self.light_wp = -1
        self.stopping_waypoints = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(2) # 10 # 40 # Hz
        while not rospy.is_shutdown():
            if self.current_pose and self.base_waypoints:
                next = self.get_next_waypoint()
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)

                if self.light_wp < 0:
                    lane.waypoints = self.base_waypoints[next : next + LOOKAHEAD_WPS]
                    self.stopping_waypoints = None
                    self.final_waypoints_pub.publish(lane)

                else:
                    self.stopping_waypoints = []
                    if next < self.light_wp:
                        for wp in self.base_waypoints[next : self.light_wp + 1]:
                            p = Waypoint()
                            wp_pose = wp.pose.pose
                            p.pose.pose.position.x = wp_pose.position.x
                            p.pose.pose.position.y = wp_pose.position.y
                            p.pose.pose.position.z = wp_pose.position.z
                            p.pose.pose.orientation = wp_pose.orientation
                            p.twist.twist.linear.x = wp.twist.twist.linear.x
                            self.stopping_waypoints.append(p)

                    lane.waypoints = self.decelerate(self.stopping_waypoints) if len(self.stopping_waypoints) > 0 else []
                    self.final_waypoints_pub.publish(lane)

            rate.sleep()

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance2(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist) * 3.6
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

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

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance2(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
