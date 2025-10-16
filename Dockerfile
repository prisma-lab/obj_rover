# =========================
# Stage 1: BUILDER
# =========================
FROM ros:humble AS builder
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# --- Repo Ignition/Gazebo keys (alcuni pacchetti dipendono da questo repo)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release ca-certificates \
 && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list

# --- Toolchain & build deps per ROS e Livox
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    python3-rosdep python3-colcon-common-extensions \
    python3-pip \
    libpcl-dev \
    # ROS deps che ti servono anche a build time
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-rviz2 \
    ros-humble-rqt* \
    ros-humble-joint-state-publisher-gui \
    ros-humble-vision-opencv \
    ros-humble-rviz-visual-tools \
    ros-humble-imu-tools \
    ros-humble-octomap-rviz-plugins \
    ros-humble-usb-cam \
    ros-humble-image-pipeline \
    ros-humble-robot-localization \
    ros-humble-xacro \
    ros-humble-tf-transformations \
    ros-humble-geometry-msgs \
    ros-humble-tf2-ros \
    ros-humble-nav-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    ignition-fortress \
    ros-humble-ros-ign-bridge \
    ros-humble-ros-gz \
    libopencv-dev \  
 && rm -rf /var/lib/apt/lists/*

# --- Compila e installa Livox-SDK2 in un prefisso dedicato (/opt/livox)
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git \
 && cd Livox-SDK2 && mkdir build && cd build \
 && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/livox \
 && make -j"$(nproc)" \
 && make install \
 && rm -rf /tmp/Livox-SDK2

# --- Workspace di build
ENV WS=/workspace/ros2_ws
RUN mkdir -p ${WS}/src
WORKDIR ${WS}

# Porta dentro il tuo codice (presuppone una cartella ./src accanto al Dockerfile)
# Se non hai nulla da copiare, commenta la riga seguente.
COPY ./src ${WS}/src

# Repo extra che usavi
WORKDIR ${WS}/src
RUN mkdir -p git && cd git \
 && git clone -b humble --single-branch https://github.com/rst-tu-dortmund/costmap_converter.git \
 && git clone -b humble-devel --single-branch https://github.com/rst-tu-dortmund/teb_local_planner.git

# Livox ROS Driver 2 (DEVE essere sotto src/)
RUN mkdir -p ${WS}/src/livox \
 && cd ${WS}/src/livox \
 && git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git

# rosdep + build (NO symlink-install, così copiamo la sola install/)
WORKDIR ${WS}
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rosdep init || true \
 && rosdep update \
 && rosdep install -i --from-path src --rosdistro ${ROS_DISTRO} -y \
      --skip-keys "OpenCV OpenCv open3d" \    # <--- AGGIUNTA
 && colcon build --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/livox


# =========================
# Stage 2: RUNTIME
# =========================
FROM ros:humble AS runtime
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=90

# --- Repo Ignition/Gazebo keys (runtime ha gli stessi pacchetti ROS)
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release ca-certificates \
 && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list

# --- Pacchetti runtime (senza toolchain pesante)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # utilità
    kmod minicom screen tmux tmuxp \
    # ROS runtime (match builder)
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-rviz2 \
    ros-humble-rqt* \
    ros-humble-joint-state-publisher-gui \
    ros-humble-vision-opencv \
    ros-humble-rviz-visual-tools \
    ros-humble-imu-tools \
    ros-humble-octomap-rviz-plugins \
    ros-humble-usb-cam \
    ros-humble-image-pipeline \
    ros-humble-robot-localization \
    ros-humble-xacro \
    ros-humble-tf-transformations \
    ros-humble-geometry-msgs \
    ros-humble-tf2-ros \
    ros-humble-nav-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    ignition-fortress \
    ros-humble-ros-ign-bridge \
    ros-humble-ros-gz \
    # Python runtime
    python3-venv python3-opencv \
 && rm -rf /var/lib/apt/lists/*

# --- Copia Livox SDK e registra libs
COPY --from=builder /opt/livox /opt/livox
RUN echo "/opt/livox/lib" > /etc/ld.so.conf.d/livox-sdk2.conf && ldconfig
ENV CMAKE_PREFIX_PATH=/opt/livox:/opt/ros/${ROS_DISTRO}
ENV LD_LIBRARY_PATH=/opt/livox/lib:${LD_LIBRARY_PATH}

# --- Copia il workspace installato dal builder
#     Lo mettiamo in /opt/ros_ws e lo sourciamo all’avvio
COPY --from=builder /workspace/ros2_ws/install /opt/ros_ws
ENV COLCON_CURRENT_PREFIX=/opt/ros_ws

# --- Utente non-root
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} user \
 && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash user \
 && echo "user:user" | chpasswd \
 && usermod -aG dialout,video,plugdev user \
 && groupadd -f iio && usermod -aG iio user
ENV HOME=/home/user
WORKDIR ${HOME}

# --- Python: virtualenv per evitare conflitti APT/pip
USER user
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Aggiorna pip nel venv
RUN python -m pip install --upgrade pip

# PyTorch CPU-only (niente ruote CUDA gigantesche) + resto pacchetti
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "numpy<2.0" \
    opencv-python opencv-contrib-python \
    transforms3d \
    Cython \
    lapx \
    open3d \
    pyrealsense2 \
    ros2-numpy \
    shapely \
    scikit-learn \
    openvino-dev \
    google-generativeai \
    ultralytics

# --- Qualche env comodo
ENV DISPLAY=:0
ENV GZ_SIM_RESOURCE_PATH=${HOME}/ros2_ws/src/ros2_iiwa/iiwa_description/gazebo/models

# --- Sourcing all’avvio
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >>  ${HOME}/.bashrc \
 && echo "source /opt/ros_ws/local_setup.bash"       >>  ${HOME}/.bashrc \
 && echo "source ${VIRTUAL_ENV}/bin/activate"         >>  ${HOME}/.bashrc

# (opzionale) copia una tmux.conf
# COPY tmux.conf ${HOME}/.tmux.conf

# --- Entrypoint
CMD ["/bin/bash"]
