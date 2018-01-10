#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np

STATE_COUNT_THRESHOLD = 1
MAX_D_SQ = 100.*100.

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_pose = None
        self.base_waypoints = None

        self.camera_image = None
        self.lights = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        #sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        self.sub_image = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1) #, buff_size=4*52428800) 
        print("subscribed")

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.stop_waypoints = []

        self.car_wp = -1
        self.nlight_i = -1
        self.nlight_wp = -1
        self.nlight = None

        self.initializing = True
        self.loop()

    def loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if not self.initializing:
                if self.current_pose and self.base_waypoints and len(self.lights) > 0 and len(self.stop_waypoints) > 0:
                    if len(self.lights) != len(self.stop_waypoints):
                      print("We have different number of stops and lights!!!")
                      break

                    # we are near this point
                    self.car_wp = self.get_closest_waypoint(self.current_pose.pose)
                    self.nlight_i = self.find_next_stop(self.car_wp)

                    self.nlight_wp = -1
                    self.nlight = None

                    if self.nlight_i >= 0 and (self.car_wp < self.stop_waypoints[self.nlight_i] + 10) and (self.car_wp > self.stop_waypoints[self.nlight_i] - 100):

                        self.nlight_wp = self.stop_waypoints[self.nlight_i]
                        self.nlight = self.lights[self.nlight_i]

                    if self.nlight_wp >= 0 and self.sub_image is None:

                        self.sub_image = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1) #, buff_size=4*52428800) 
                        print("subscribed")

                    elif self.nlight_wp < 0 and self.sub_image is not None:

                        self.sub_image.unregister()
                        self.sub_image = None
                        print("unsubscribed")
                        self.last_wp = -1
                        self.upcoming_red_light_pub.publish(Int32(-1))

                else:
                    print("No pose, or no waypoints, or # of lights or stops is zero!!!")
                    break

            rate.sleep()

    def pose_cb(self, msg):
        self.current_pose = msg

    def waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints;

        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_positions = self.config['stop_line_positions']

        for stop_pos in self.stop_positions:
           stop_pose_stamped = self.create_dummy_pose(stop_pos[0], stop_pos[1], 0)
           stop_wp = self.get_closest_waypoint(stop_pose_stamped.pose)
           self.stop_waypoints.append(stop_wp)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1

            if state == TrafficLight.RED:
              print(("nlight_i:", self.nlight_i, "RED", self.state_count, "car_wp:", self.car_wp, "nlight_wp:", self.nlight_wp))
            if state == TrafficLight.YELLOW:
              print(("nlight_i:", self.nlight_i, "YELLOW", self.state_count, "car_wp:", self.car_wp, "nlight_wp:", self.nlight_wp))

            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

        if self.initializing:
            self.initializing = False

    def create_dummy_pose(self, x, y, z):
        pose = PoseStamped()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        return pose

    def find_next_stop(self, wp):
        for i in range(len(self.stop_waypoints)):
           if wp < self.stop_waypoints[i] + 10:
              return i
        return -1

    def distance_sq(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return x*x + y*y + z*z

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.base_waypoints

        """
        self_pos = pose.position
        distances = [self.distance_sq(self_pos, way_pos.pose.pose.position) for way_pos in self.base_waypoints]
        return np.argmin(distances)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.initializing:
            state = self.get_light_state(0)
            return -1, TrafficLight.UNKNOWN

        if self.nlight_wp and self.nlight:
            return self.nlight_wp, self.nlight.state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
