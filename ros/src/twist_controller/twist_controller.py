
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
KP = 2.0
KI = 0.4
KD = 0.1
TAU = 1.0
TS = 1.0

class TwistController(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, \
                       wheel_radius,  wheel_base, steer_ratio, max_lat_accel, max_steer_angle, min_speed):

        self.vehicle_mass    = vehicle_mass   
        self.fuel_capacity   = fuel_capacity  
        self.brake_deadband  = brake_deadband  
       	self.decel_limit     = decel_limit    
        self.accel_limit     = accel_limit    
        self.wheel_radius    = wheel_radius   
        self.wheel_base      = wheel_base     
        self.steer_ratio     = steer_ratio    
        self.max_lat_accel   = max_lat_accel  
        self.max_steer_angle = max_steer_angle
        self.min_speed       = min_speed

        self.full_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY

        self.enabled = False

        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, self.min_speed, self.max_lat_accel, self.max_steer_angle)
        self.lowpass_steer = LowPassFilter(TAU, TS)
        self.lowpass_accel = LowPassFilter(TAU, TS)
        self.pid = PID(kp=KP, ki=KI, kd=KD, mn=self.decel_limit, mx=self.accel_limit)

    def enable(self, enable):
        if enable and not self.enabled:
           self.enabled = True
           self.pid.reset()
        elif not enable and self.enabled:
           self.enabled = False
           self.pid.reset()
        else:
           pass

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_velocity, dbw_enabled, dt):

        target_linear_velocity = proposed_linear_velocity.x
        target_angular_velocity = proposed_angular_velocity.z
        current_linear_velocity = current_velocity.twist.linear.x

        acceleration = self.pid.step(target_linear_velocity - current_linear_velocity, dt)
        acceleration = self.lowpass_accel.filt(acceleration)

        if acceleration > 0.0:
            throttle = acceleration
            brake = 0.0
        else:
            throttle = 0.0
            if acceleration > self.brake_deadband:
                brake = 0.0
            else:
                brake = -acceleration*self.full_mass*self.wheel_radius

        steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
        steer = self.lowpass_steer.filt(steer)

        # Return throttle, brake, steer
        return throttle, brake, steer
