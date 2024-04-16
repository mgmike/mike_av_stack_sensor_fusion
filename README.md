# Mike AV Stack - Sensor Fusion

## About the project

The sensor fusion part of my av stack includes object detection from camera and lidar scanners. 

The main algorithms such as tracking, Extended Kalman Filters, CNNs... are inspired heavily from the Udacity Self driving car nanodegree. I have changed the code significantly as I am adapting it to ros2 foxy inside a docker container. I have also made the detection asynchronous as the ros topics will produce messages at varying intervals. 


## Prerequisites

Ubuntu 20.04\
Mike av stack base

### Prerequisites: Local 
ros2 foxy\
opencv\
pytorch\
(vscode recommended)


### Prerequisites: Docker
Docker

Make sure you have the correct drivers and docker versions for nvidia integration
https://docs.nvidia.com/ai-enterprise/deployment-guide-vmware/0.1.0/docker.html

Add nvidia runtime
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/user-guide.html

## Installation
### Installation: Local

1. Clone the repo from github into the source folder of your ament workspace

    For example, you may want your workspace to look like 
    ```
    workspace_folder/
        src/
            package1/
                ...
            mike_av_stack/
                ...
            mike_av_stack_sensor_fusion/ 
                ...
    ```
    ```sh
    $ cd .../workspace_folder/src/
    $ git clone https://github.com/mgmike/mike_av_stack_sensor_fusion.git
    ```
3. Move into workspace directory and build
    ```sh
    $ cd ../
    $ pip install -r src/mike_av_stack_sensor_fusion/.devcontainer/requirements.txt
    $ source /opt/ros/foxy/setup.bash
    $ colcon build --packages-select mike_av_stack_sensor_fusion
    $ source install/setup.bash
    ```

### Installation: Docker

#### No vscode

```sh
$ docker build . -t ws_av_sf -f .devcontainer/Dockerfile
$ docker run -it -p 8888:8888 -v ./:/home/ws_av_sf/ros2_ws/src/mike_av_stack_sensor_fusion -v /dev/shm:/dev/shm -e DISPLAY=0 -e NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia --env="DISPLAY" --gpus all ws_av_sf
```

#### With VScode

Open the mike_av_stack_sensor_fusion directory in vscode.

```
$ code src/mike_av_stack_sensor_fusion/
```
 
Use `View->Command Palette` then search for `Dev Containers: (Re-)build and Reopen in Container`. This will build the docker image and run the container inside the vscode environment. 

Open a new terminal using `View->Terminal` or `Ctrl+Shift+` and `New Terminal`

Once inside, run the following:
```sh
$ cd ../../
$ pip install -r src/mike_av_stack_sensor_fusion/.devcontainer/requirements.txt
$ source /opt/ros/humble/setup.bash
$ colcon build --packages-select mike_av_stack_sensor_fusion
$ source install/setup.bash
```


## Running the node:

Start Carla (Host machine)
```
$ cd /opt/carla-simulator/
$ ./CarlaUE4.sh
```

Start mike av stack base (Host machine)
```
$ cd .../workspace_folder/
$ export ROS_DOMAIN_ID=1
$ source install/setup.bash
$ ros2 launch mike_av_stack av_sf.launch.py
```

Start mike av stack sensor fusion node
```
ros2 run mike_av_stack_sensor_fusion sensor_fusion --ros-args --log-level debug
```


<!-- 
The following may be needed in the next step: Uisng yolov8 instead of resnet
$ export PYTHONPATH=$PYTHONPATH:/home/ws_av_sf/.local/lib/python3.10/site-packages
$ export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
 -->