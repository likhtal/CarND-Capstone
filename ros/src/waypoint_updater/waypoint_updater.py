#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import tf

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        # current path
        # NOTE: supposedly comes from top level planner (from file for simulator) at rate 40Hz
        self.current_waypoints = None

        # current pose     
        # NOTE: supposedly comes from fusion (briged from simulator) at unknown rate   
        self.current_pose = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(10) # 40 # Hz
        while not rospy.is_shutdown():
            if ((self.current_pose is not None) and (self.current_waypoints is not None)):
                next_waypoint_index = self.get_next_waypoint()
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = self.current_waypoints[next_waypoint_index:next_waypoint_index+LOOKAHEAD_WPS]
                self.final_waypoints_pub.publish(lane)

            rate.sleep()

    def euclidean_distance(self, position1, position2):
        a = position1
        b = position2
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

    def get_closest_waypoint(self):
        min_dist = 100000
        min_ind = 0
        ind = 0
        position1 = self.current_pose.pose.position
        for wp in self.current_waypoints:
            position2 = wp.pose.pose.position 
            dist = self.euclidean_distance(position1, position2)
            if dist < min_dist:
                min_dist = dist
                min_ind = ind
            ind += 1
        return min_ind

    def current_yaw(self):
        quaternion = (self.current_pose.pose.orientation.x)
        quaternion = (
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def get_next_waypoint(self):
        ind = self.get_closest_waypoint()
        
        map_x = self.current_waypoints[ind].pose.pose.position.x
        map_y = self.current_waypoints[ind].pose.pose.position.y

        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y

        heading = math.atan2((map_y-y), (map_x-x))
        yaw = self.current_yaw()
        angle = abs(yaw - heading);

        #print ('yaw:', yaw, 'heading:', heading, 'angle:', angle, map_x, map_y, x, y)

        if (angle > math.pi/4):
            ind += 1

        #print ('finx:', self.current_waypoints[ind].pose.pose.position.x, 'finy:', self.current_waypoints[ind].pose.pose.position.y)
        return ind

    def pose_cb(self, msg):
        self.current_pose = msg
        pass

    def waypoints_cb(self, lane):
        self.current_waypoints = lane.waypoints;
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
