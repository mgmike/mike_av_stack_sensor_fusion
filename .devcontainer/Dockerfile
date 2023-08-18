FROM ros:foxy
ARG USERNAME=ws_av_sf
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# Copy code over
WORKDIR /home/ws_av_sf/src/mike_av_stack_sensor_fusion/
COPY  . .

# COPY requirements.txt /tmp/

# Install requirements
# RUN pip install easydict
# RUN pip install torchvision
# RUN pip install opencv-python
# RUN pip install open3d



# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

RUN echo "ls -la"

RUN sudo apt-get install -y ros-foxy-vision-msgs python3-opencv
# RUN pip install -r requirements.txt
# RUN pip3 install open3d>=0.17.0

CMD ["/bin/bash"]

