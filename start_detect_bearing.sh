#!/bin/bash
roslaunch usb_cam usb_all_cam.launch & sleep 3;
docker exec -it detect bash -c "cd /home/detect_ws && source devel/setup.bash && roslaunch target_dist_calculator target_bearing.launch &  sleep 2 &&  cd /home/detect_ws && source devel/setup.bash && roslaunch py_yolov8 yolov8.launch";
wait
