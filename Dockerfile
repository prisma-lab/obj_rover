FROM ros:humble

#Uncomment the following line if you get the "release file is not valid yet" error during apt-get
#	(solution from: https://stackoverflow.com/questions/63526272/release-file-is-not-valid-yet-docker)
#RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until

RUN echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections

#Install essential
RUN apt-get update && apt-get install -y
RUN apt-get install software-properties-common dialog apt-utils apt-transport-https curl -y
RUN apt-get install ros-humble-turtlesim -y
RUN apt-get install -y lsb-release gnupg
RUN sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install -y ignition-fortress

RUN apt-get update && apt-get install -y

##You may add additional apt-get here
RUN apt install kmod -y
RUN apt install ros-humble-rviz2 -y
RUN apt install ros-humble-rqt* -y
RUN apt install minicom -y
RUN apt install screen -y
RUN apt install ros-humble-xacro -y
RUN apt install ros-humble-librealsense2* -y
RUN apt install ros-humble-realsense2-* -y
RUN apt install ros-humble-navigation2 -y
RUN apt install ros-humble-nav2-bringup -y
RUN apt install ros-humble-slam-toolbox -y
RUN apt install ros-humble-turtlebot3-gazebo -y
RUN apt install ros-humble-vision-opencv -y
RUN apt install ros-humble-rmw-cyclonedds-cpp -y
RUN apt install ros-humble-joint-state-publisher-gui -y

RUN apt-get update && apt-get install -y
RUN apt install ros-humble-rtabmap -y
RUN apt install ros-humble-rtabmap-ros -y
RUN apt install ros-humble-rviz-visual-tools -y
RUN apt install ros-humble-imu-tools -y
RUN apt install ros-humble-octomap-rviz-plugins -y

#JOYSTICK
RUN apt install joystick -y
RUN apt install ros-humble-joy -y
RUN apt install ros-humble-teleop-twist-joy -y

#install for gz sim
RUN apt-get install -y ros-humble-ros-ign-bridge && \
apt-get install -y ros-humble-ros-gz -y
RUN apt-get install ros-humble-controller-manager -y
RUN apt-get install ros-humble-ros2-control -y
RUN apt-get install ros-humble-ros2-controllers -y
RUN apt-get install ros-humble-ign-ros2-control -y
RUN apt-get install ros-humble-ign-ros2-control-demos -y

RUN apt-get install ros-humble-usb-cam -y
RUN apt-get install ros-humble-image-pipeline -y
RUN apt-get install ros-humble-tf-transformations -y

RUN apt-get upgrade -y && apt-get update -y

RUN apt-get install ros-humble-robot-localization -y

RUN apt-get update && apt-get install -y
RUN sudo apt install pip -y
RUN pip3 install opencv-python opencv-contrib-python
RUN pip3 install --upgrade transforms3d
RUN apt-get update && apt install ros-humble-tf-transformations -y
RUN pip3 install --no-cache-dir Cython

# Installa torch PRIMA di ultralytics per evitare conflitti
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir lapx ultralytics open3d pyrealsense2 ros2-numpy shapely scikit-learn openvino-dev

RUN pip3 install -q -U google-generativeai

RUN pip3 install "numpy<2.0"

# Installaazione per CLIP

RUN pip3 install ftfy regex tqdm

RUN pip3 install git+https://github.com/openai/CLIP.git

#RUN export GZ_SIM_RESOURCE_PATH=~/ros2_ws/src/ros2_iiwa/iiwa_description/gazebo/models
ENV GZ_SIM_RESOURCE_PATH=~/ros2_ws/src/ros2_iiwa/iiwa_description/gazebo/models

#Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV HOME=/home/user
ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=90

# Installazione Livox-SDK2
# Questo deve essere fatto come root prima di creare l'utente
RUN apt-get update && apt-get install -y cmake git build-essential
WORKDIR /tmp
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd Livox-SDK2 && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf Livox-SDK2

# Aggiungi il percorso delle librerie Livox al sistema
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/livox.conf && ldconfig

# Imposta la variabile d'ambiente per le librerie condivise
ENV LD_LIBRARY_PATH=/usr/local/lib

#Add non root user using UID and GID passed as argument
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN echo "user:user" | chpasswd
# Consenti a user di usare sudo senza password (necessario per rosdep)
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN sudo adduser user dialout
RUN sudo adduser user video
RUN sudo adduser user sgx
RUN sudo groupadd iio
RUN sudo adduser user iio
RUN sudo adduser user plugdev

USER user

#ROS2 workspace creation
RUN mkdir -p ${HOME}/ros2_ws/src
WORKDIR ${HOME}/ros2_ws
COPY --chown=user cyclonedds_config.xml ${HOME}/cyclonedds_config.xml
ENV CYCLONEDDS_URI=file://${HOME}/cyclonedds_config.xml
COPY --chown=user ./src ${HOME}/ros2_ws/src
SHELL ["/bin/bash", "-c"] 
WORKDIR ${HOME}/ros2_ws/src/git

## Clone the required repositories first
#RUN git clone -b humble --single-branch https://github.com/ros-perception/vision_opencv.git
RUN git clone -b humble --single-branch https://github.com/rst-tu-dortmund/costmap_converter.git
RUN git clone -b humble-devel --single-branch https://github.com/rst-tu-dortmund/teb_local_planner.git

# Return to the workspace root
WORKDIR ${HOME}/ros2_ws

# Clone Livox ROS Driver 2 in un workspace separato e compilalo con il suo build.sh
WORKDIR /tmp
RUN git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2 && \
    cd ws_livox && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    chmod +x src/livox_ros_driver2/build.sh && \
    ./src/livox_ros_driver2/build.sh humble

# Copia i file installati di livox nel workspace principale
RUN mkdir -p ${HOME}/ros2_ws/install/livox_ros_driver2 && \
    cp -r /tmp/ws_livox/install/livox_ros_driver2/* ${HOME}/ros2_ws/install/livox_ros_driver2/ && \
    rm -rf /tmp/ws_livox

# Torna al workspace principale
WORKDIR ${HOME}/ros2_ws

# Esegui rosdep per installare le dipendenze dei pacchetti rimanenti, poi compila con colcon
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep update && \
    rosdep install -i --from-path src --rosdistro humble -y \
        --skip-keys="livox_ros_driver2 open3d aruco_pose_estimation yolov11_ros2 python3-torch torch python3-torchvision torchvision python3-torchaudio torchaudio ultralytics" && \
    colcon build --symlink-install \
        --cmake-args \
        -DCMAKE_BUILD_TYPE=Release

# Aggiungi il source del workspace al bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash;" >> ${HOME}/.bashrc
RUN echo "source ${HOME}/ros2_ws/install/setup.bash;" >> ${HOME}/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:\${LD_LIBRARY_PATH}" >> ${HOME}/.bashrc
RUN echo 'export CYCLONEDDS_URI=file://$HOME/cyclonedds_config.xml' >> ${HOME}/.bashrc

#Clean image
USER root
# Additional packages
RUN apt update && apt install -y --no-install-recommends \    
    aptitude \
    tmux \
    tmuxp 

COPY tmux.conf .tmux.conf
RUN rm -rf /var/lib/apt/lists/*
USER user
