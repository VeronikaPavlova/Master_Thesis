import time

import roslaunch
import rospkg
import rospy
import getch

from ros_labjack.msg import Measurements


def main():
    rospy.init_node("StrainSensor", anonymous=False)

    experimentLoop()
    # Get available ROS topics
    # ros_topics = [top[0] for top in rospy.get_published_topics()]
    # rospy.Subscriber('/sensordata/finger_1', Measurements, strain_sensor_callback)

    # Python should not exit node
    rospy.spin()

def experimentLoop():
    print("Start [Experimental Loop]")
    looping = True
    print("Start Strain Sensors [Y/n]")
    char = getch.getch()
    if char.lower() in [" ", "/n", "y"]:
        startStrainSensorNode()
        time.sleep(2)

    print("\n ============================= \n Press <ANY KEY> to start! \n ============================= \n")
    char = getch.getch()
    while looping:
        print("print Space to start recording!")
        char = getch.getch()

        while char != " ":
            if char == "q":
                print("Quitting Experiment")
                return

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
