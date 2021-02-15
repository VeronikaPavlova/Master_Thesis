#!/usr/bin/env python

import roslaunch
import rostopic

topiclist = rostopic.get_topic_list()


package = 'rqt_gui'
executable = 'rqt_gui'
node = roslaunch.core.Node(package, executable)

launch = roslaunch.scriptapi.ROSLaunch()
launch.start()

process = launch.launch(node)
print(process.is_alive())
# process.stop()