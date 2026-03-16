FROM ros:humble-ros-base-jammy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/ros2
COPY ros2 /app/ros2

RUN rosdep update && \
    apt-get update && \
    rosdep install --from-paths src --ignore-src -y && \
    apt-get install -y python3-matplotlib python3-pandas && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc
