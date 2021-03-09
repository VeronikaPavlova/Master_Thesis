import os
import sys
import time

import roslaunch
import rospkg
import rospy
import getch

from ros_labjack.msg import Measurements

BASE_DIR = "../.."
MODEL_NAME = "Test"

def main():
    # rospy.init_node("StrainSensor", anonymous=False)

    global DATA_DIR
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME)

    experimental_loop()
    # Get available ROS topics
    # ros_topics = [top[0] for top in rospy.get_published_topics()]
    # rospy.Subscriber('/sensordata/finger_1', Measurements, strain_sensor_callback)

    # Python should not exit node
    # rospy.spin()


def experimental_loop():
    print("Start [Experimental Loop]")
    looping = True
    print("Start Strain Sensors")
    # char = getch.getch()
    # if char.lower() in [" ", "/n", "y"]:
    startStrainSensorNode()
    time.sleep(2)


    while looping:
        print("Space to start recording!")
        char = getch.getch()

        while char != " ":
            if char == "q":
                print("Quitting Experiment")
                return
        start_recording()
        time.sleep(2)

        # Do some stuff

        print("Press Space when you are done!")
        char = getch.getch()
        while char != " ":
            if char == "q":
                print("Quitting Experiment")

                return

def start_recording():

    # Get available ROS topics
    ros_topics = [top[0] for top in rospy.get_published_topics()]
    print(ros_topics)

    rosbagFolder = os.path.dirname(sys.argv[0]) + "/" + MODEL_NAME + "/rosbag"

    package = 'rosbag'
    executable = 'record'

    if not os.path.exists(rosbagFolder):
        os.makedirs(rosbagFolder)

    prefix = "test"
    rosbagName = rosbagFolder + os.sep + prefix
    node = roslaunch.core.Node(package, executable,
                               args="-e '(/sensordata/finger)' -o {}".format(rosbagName))

    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    global rosbagProcess
    rosbagProcess = launch.launch(node)
    rosbagProcess.rosbagName = rosbagName

    print(" .... recording started!")

def stop_recording():
    global rosbagProcess
    rosbagProcess.stop()

def mkpath(*args):
    """ Takes parts of a path (dir or file), joins them, creates the directory if it doesn't exist and returns the path.
        figure_path = mkpath(PLOT_DIR, "experiment", "figure.svg")
    """
    path = os.path.join(*args)
    if os.path.splitext(path)[1]:  # if path has file extension
        base_path = os.path.split(path)[0]
    else:
        base_path = path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return path

def startStrainSensorNode():

    # Start Strain Sensor Node from launch file
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    rospack = rospkg.RosPack()
    sensorLaunchfile = rospack.get_path("ros_labjack") + "/launch/twoComp_sensors.launch"

    global sensorLaunch
    sensorLaunch = roslaunch.parent.ROSLaunchParent(uuid, [sensorLaunchfile])

    sensorLaunch.start()


def strain_sensor_callback(data):
    # Each subscriber gets 1 callback, and the callback either
    # stores information and/or computes something and/or publishes
    # It _does not!_ return anything
    print(data.values)


if __name__ == "__main__":
    main()
