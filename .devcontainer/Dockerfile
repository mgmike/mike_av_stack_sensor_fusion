FROM osrf/ros:humble-desktop


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# Set up user as root
ARG USERNAME=ws_av_sf
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# RUN echo 'root:root' | chpasswd

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG sudo $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install packges for project
RUN apt-get update -q && \
    apt-get install -y ros-humble-vision-msgs ros-humble-derived-object-msgs ros-humble-ament-cmake-nose python3-pip
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Copy code over
# WORKDIR /home/$USERNAME/ros2_ws/src/mike_av_stack_sensor_fusion/
# COPY  . .

# WORKDIR /home/ws_av_sf/src/
# RUN git clone --branch foxy-devel https://github.com/Box-Robotics/ros2_numpy.git

WORKDIR /home/$USERNAME/ros2_ws/

CMD ["/bin/bash"]

