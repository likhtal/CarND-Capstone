## Submitter Info

**THIS IS AN INDIVIDUAL PROJECT - I AM NOT A MEMBER OF ANY TEAM, SO I DON'T EXPECT THIS TO RUN ON CARLA.**

## Project Description

This project attempts to control a car in Udacity simulator by creating trajectory for the car, responding to the traffic lights status etc. The car in simulator is being controlled by a few ROS nodes, which in turn run in Ubuntu 16.04.

### Installation/Set-up

Originaly, I was using the following configuration:

* My machine: Surface Book Pro, Intel Core I7-6600U at 2.60GHz 2.81 GHz, with 16GB of RAM, Nvidia GPU with 1GB of memory, running Windows 10 Pro.
* For running ROS and nodes I was using VirtualBox with VM from Udacity [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/7e3627d7-14f7-4a33-9dbf-75c98a6e411b/concepts/8c742938-8436-4d3d-9939-31e40284e7a6?contentVersion=1.0.0&contentLocale=en-us).
* I had some minor issues with enablig screen resizing for the VM. This [link](https://askubuntu.com/questions/321589/unable-to-mount-the-cd-dvd-image-on-the-machine-sandbox) helped me.
* I had to increase the harddrive size for the VM. Those links helped: [here](https://stackoverflow.com/questions/11659005/how-to-resize-a-virtualbox-vmdk-file)
  and [here](https://tvi.al/resize-sda1-disk-of-your-vagrant-virtualbox-vm/).
* I ran the simulator in Windows 10 Pro (that is, in the host OS).

The configuration proved to be **sadly inadequate performance-wise** - see "Performance Issues" below. 

### Performance Issues

After I had implemented Waypoint Updater Node (Partial) with camera turned off - so the car would drive smoothly around the loop, I turned on the camera.
I found immediately that the car would lose the track as soon as the camera is on - even if I am not subscribing to any images - just publication would be enough.
In order to overcome this obstacle I implemented all trick I found from my colleagues-students - see, for example, adjusting rates, and also implementing unsubscribe from image topic when the car is not close to any traffic light (essentially, borrowing it from [here](https://github.com/diyjac/SDC-System-Integration)).

Nothing was helping much - then I came up with the idea of dropping all but every nth image, and it did help - see bridge.py:

```python
EVERY_NTH_IMAGE = 6
...
    def publish_camera(self, data):
        if self.n == 0:
           imgString = data["image"]
           image = PIL_Image.open(BytesIO(base64.b64decode(imgString)))
           image_array = np.asarray(image)
           image_message = self.bridge.cv2_to_imgmsg(image_array, encoding="rgb8")
           self.publishers['image'].publish(image_message)
           #print((rospy.get_time(), "image"))

        self.n += 1
        self.n = self.n % EVERY_NTH_IMAGE
```

After that the car was able to make it around the loop with the camera on. Until Microsoft aplied their update for Meltdown/Spectre to my machine.
Everything stopped working immediately. Apparently, the slowdown caused by the update was enough to break the balance. I had to roll back the update and freeze Microsoft updates till I submit the project.

Additional issue was/is using Faster RCNN model, essentially following the work of Cold Knight [here](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI).
I set up P2 instance in AWS to train the models and succeeded. Unfortunately, even after all optimizations, running detection in VM takes a few seconds per image.
(Running detection on my machine using my GPU takes about half a second per image; running detection in P2 takes about 160 msecs - which would be acceptable with image publishing rate at 1-2 images per second.)

In the end I tried [paperspace](paperspace.com) approach with Docker GPU, and that worked really well. ML-In-aBox with Ubuntu 16.04 was a perfect fit - except that, eventually, I had to upgrade to 100GB of HD (from 50 GB).
50GB was enough for everything, except for movie creation.

### Paperspace/Docker GPU

Instructions for installation (and running) Docker GPU are located [here](github.com/team-inrs/CarND-Capstone). The process generated docker with the team content, hich I overwrote (under ros/) with mine.

On paperspace I have it installed at: ~/capstone/CarND-Capstone  (ros-i/ is team-inrs, ros/ is mine)

I set up AWS CLI in the docker, to get my code from AWS S3.

One caveat: I needed to assign execution permissions to all .py, .sh files (having been copied from Windows, then S3).

Downloaded simulator to: ~/sim/linux_sys_int/sys_int.x86_64

To run:

1. From ~/capstone/CarND-Capstone/, run sudo ./run-cuda.sh

Got into docker, base path is /udacity

2. cd ros

3. catkin_make OR catkin_make clean, catkin_make

4. source devel/setup.bash

5. roslaunch launch/styx.launch

6. run simulator

I was using [recordmydesktop](https://askubuntu.com/questions/339207/how-to-install-recordmydesktop-in-ubuntu) to record videos - e.g. https://www.youtube.com/watch?v=oU7trx-vujs

### The logic

The logic of nodes interaction is described by Udacity [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/455f33f0-2c2d-489d-9ab2-201698fbf21a).

What represents certain interest is the logic I implemented for handling traffic lights information. 

I am reporting red **or** yellow light as red to the waypoint_updater node, as well as the following information whether we have camera on yet or no.

The logic goes as follows:

* if there is no camera yet we just stay still where we are:

```python
                if self.light_wp == NO_CAMERA_YET:  # no camera
                    self.state = State.NoCameraYet
                    lane.waypoints = self.no_go()
```

* if the camera is on and there is no traffic light nearby, or there is, but the light is green, we are moving forward:

```python
                elif self.light_wp == NO_RED:  # free to go
                    self.state = State.Going
                    lane.waypoints = self.just_go(next)
```

* if we have red or yellow light and we are in slow start mode - that is, whether we stay before the very first stop line, then we slowly move to our first stop light (and switch to "end of slow start" state when we are done):

```python
                else:  # red light found
                    if self.state == State.NoCameraYet or self.state == State.SlowStart:  # slow starting, with camera
                        self.state = State.SlowStart   # slow-start
                        lane.waypoints = self.slow_down(next, self.light_wp, slow_start=True)
                        if len(lane.waypoints) <= 0:
                            self.state = State.EndOfSlowStart
```

* if we are staying at the first stop line ("end of slow start") we just stay there.

```python
                    elif self.state == State.EndOfSlowStart:  # end of slow start
                        self.state = State.EndOfSlowStart
                        lane.waypoints = self.no_go()
```

* if we are going or stopping, and we see the red/yellow light, and we did not reach stop line yet - so we have plenty of time to stop - we start slowing down to stop:

```python
                    elif next < self.light_wp:  # going or stopping, and did not reach stop line yet
                        self.state = State.Stopping  # stopping
                        lane.waypoints = self.slow_down(next, self.light_wp)
```

* we have red/yellow light, we are at the stop line (or past it), and we are stopping already - just stop.

```python
                    elif self.state == State.Stopping:  # stopping already, doing nothing
                        self.state = State.Stopping
                        lane.waypoints = self.no_go()
```

* we have red/yellow light, we are at the stop line (or past it) and we are going full speed - hmmm... it seems the light just switched to yellow - let's go fast.

```python
                    else:  # going past red light: apparently a sudden switch right under the light, continuing going
                        self.state = State.Going
                        lane.waypoints = self.just_go(next)
```

### Traffic Lights Detection

I was using/reproducing approach from https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI

Created a P2 AWS instance, and use Putty with enabled Jupyter support for it.

After installation: go to: prj/tensorflow/models/research

From models/research/: export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

To test, run python object_detection/builders/model_builder_test.py

Pre-trained models were updated since, so folders names for the models are different now. Also, there were some minor issues with the Google python code that I had to fix (version difference, I guess).

Cold Knight wrote "Due to some unknown reasons the model MobileNet SSD v1 gets trained but does not save for inference. Ignoring this for now." - it worked for me, though.

I used Faster R-CNN and MobileNet SSD. MobileNetworks much-much faster, and gives really poor quality, so I had to stick with Faster R-CNN (and paperspace, b/c otherwise detection was taking to long.)

### Other

For twist controller I am using pretty standard PID with pretty standard values (I've seen them used by multiple students), and yaw controller provided by Udacity.
I am resetting PID controller if dbw_enabled value changes.

```python
self.controller.enable(self.dbw_enabled)
```

TL Detector by default uses information on traffic lights provided by Udacity. Unfortunately, my laptop performance is not good enough to use the classifier by default.

Yet tl_detector.launch file contains a key that controls use of classifier: use_classifier - see below:

```xml
<?xml version="1.0"?>
<launch>
    <node pkg="tl_detector" type="tl_detector.py" name="tl_detector" output="screen" cwd="node">
        <param name="model_path" value="$(find styx)../../../classifier_sim/frozen_inference_graph.pb" />
        <param name="use_classifier" value="False" />
    </node>
</launch>
```

If use_classifier has value "True", the app will use Faster RCNN classifier. Works really well with paperspace.

I've updated sim_traffic_light_config.yaml to have actual coordinates for **stop lines** (not traffic lights!), so the car could properly stop at the stop line, rather than going all the way under the traffic light.

## UDACITY INSTRUCTIONS BELOW

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
