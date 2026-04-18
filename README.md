# ROS Mapping & Object Search README

## Overview

This project integrates:

- Manual mapping using keyboard control
- Autonomous navigation
- YOLO-based object search
- USB camera input

It uses two Python scripts:

- `auto_mapper.py` → for manual SLAM mapping and map saving  
- `yolo_object_search.py` → for autonomous object detection and navigation  

---

## 1. Setup

### Place scripts into your ROS package

```bash
mv auto_mapper.py ~/catkin_ws/src/<your_package>/scripts/
mv yolo_object_search.py ~/catkin_ws/src/<your_package>/scripts/
```

### Make them executable

```bash
chmod +x ~/catkin_ws/src/<your_package>/scripts/auto_mapper.py
chmod +x ~/catkin_ws/src/<your_package>/scripts/yolo_object_search.py
```

### Build workspace

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

## 2. Required ROS Packages

Make sure the following are installed:

- usb_cam
- gmapping (or your SLAM package)
- move_base
- amcl
- map_server
- darknet_ros
- tf
- actionlib

---

## 3. Launch Sequence

Open multiple terminals and run the following in order.

Always source ROS in every terminal:

```bash
source ~/catkin_ws/devel/setup.bash
```

---

## 4. Start USB Camera

```bash
roslaunch <camera_package> usb_cam.launch
```

---

## 5. Start Mapping (SLAM)

```bash
roslaunch <mapping_package> mapping.launch
```

---

## 6. Run Manual Mapper

```bash
rosrun <your_package> auto_mapper.py
```

Controls:

W/↑ forward  
S/↓ backward  
A/← left  
D/→ right  
SPACE stop  
+ faster  
- slower  
M save map  
Q save and quit  

---

## 7. Map Saving

Saved to:

~/demo_map.yaml  
~/demo_map.pgm  

Manual save:

```bash
rosrun map_server map_saver -f ~/demo_map
```

---

## 8. Start Navigation

```bash
roslaunch <navigation_package> navigation.launch
```

---

## 9. Run YOLO Object Search

```bash
rosrun <your_package> yolo_object_search.py
```

---

## 10. Target Object

Default: bottle  

Change:

```bash
rosrun <your_package> yolo_object_search.py _target_class:=person
```

---

## 11. Full Workflow

1. roslaunch usb_cam  
2. roslaunch mapping  
3. rosrun auto_mapper.py  
4. save map  
5. roslaunch navigation  
6. rosrun yolo_object_search.py  

---

## 12. Notes

- Mapping → Navigation → Object Search order matters  
- Use RViz for pose and goals  
- Ensure YOLO is publishing detections  
