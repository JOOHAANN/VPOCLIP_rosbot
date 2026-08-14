# CLIPGCN on ROSbot

This project runs CLIPGCN human action recognition (HAR) on ROSbot camera
frames under ROS 2 Jazzy. The same YOLO detection used by HAR can publish the
selected person to the person-follow controller. A velocity arbiter coordinates
Nav2 and person following with this fixed priority:

```text
Emergency stop > Nav2 navigation > Person following > Stop
```

The runtime data flow is:

```text
ROSbot camera -> CLIPGCN / YOLO -> HAR result and PersonDetection
                                          |
                                          v
                                  person_follow_controller
                                          |
                                          v
Nav2 -> cmd_vel_nav -----------> cmd_vel_arbiter -> ROSbot cmd_vel
Person follow -> cmd_vel_follow /
```

## Repository and workspace layout

The commands in this guide use the current workstation layout:

```text
/home/youhan/HAR/CLIPGCN             HAR source code and configuration
/home/youhan/HAR/VPOCLIP_rosbot      Docker Compose configuration
/home/youhan/HAR/person_follow_ws    Person-follow ROS 2 workspace
/home/youhan/HAR/X3D                 X3D source code
/home/youhan/HAR/CTR-GCN_17          CTR-GCN source code
/home/youhan/HAR/yolov5              YOLOv5 source code
```

Model files, datasets, and training outputs are excluded by `.gitignore` and
are not stored on GitHub. Restore at least these local assets before running
the full HAR pipeline:

```text
data/contrastive_zsl_splits/50_5/
work_dir/clipgcn_contrastive_50_5/run_20260616_210139/swa_model.pth
local_models/
```

## 1. Configure and verify Wi-Fi

The current DDS configuration uses wireless interface `wlp2s0`, ROS Domain ID
`10`, and peer addresses `10.186.13.100`, `10.186.13.101`, and
`10.186.13.30`.

List network devices and available Wi-Fi networks:

```bash
nmcli device status
nmcli device wifi list ifname wlp2s0
```

Connect using an existing NetworkManager profile:

```bash
nmcli connection show
nmcli connection up id "YOUR_ROSBOT_WIFI_CONNECTION"
```

To connect for the first time:

```bash
ROSBOT_WIFI_SSID="YOUR_WIFI_SSID"
ROSBOT_WIFI_PASSWORD="YOUR_WIFI_PASSWORD"
nmcli device wifi connect "$ROSBOT_WIFI_SSID" \
  password "$ROSBOT_WIFI_PASSWORD" \
  ifname wlp2s0
unset ROSBOT_WIFI_PASSWORD
```

Check the workstation address and robot connectivity:

```bash
ip -4 -br address show wlp2s0
ping -c 1 10.186.13.100
ping -c 1 10.186.13.101
```

If the wireless interface is not named `wlp2s0`, update the CycloneDDS
`NetworkInterface name` in
`/home/youhan/HAR/VPOCLIP_rosbot/compose.yaml`. Also update the DDS interface
used by the `jazzy-rosbot` navigation container. Recreate the inference
container after changing its Compose configuration.

Keep Wi-Fi credentials in NetworkManager or your local shell. Never commit
them to this repository.

## 2. Start the containers

Allow the containers to display OpenCV and RViz windows, then start the two
existing containers:

```bash
xhost +si:localuser:root
docker start jazzy-rosbot jazzy-rosbot-clipgcn
docker ps --filter name=jazzy-rosbot
```

Use the following command only for the first creation, or after changing the
Dockerfile, requirements, or Compose configuration:

```bash
cd /home/youhan/HAR/VPOCLIP_rosbot
docker compose up -d --build
```

Do not use `docker compose down` for a routine stop because it deletes the
inference container. Stop the existing containers with:

```bash
docker stop jazzy-rosbot-clipgcn jazzy-rosbot
```

## 3. Select rosbot_1 or rosbot_2

Every new Follow or HAR terminal must source the robot selector. Select robot
1 with:

```bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
```

Select robot 2 with:

```bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_2
```

The selector exports:

```text
ROSBOT_TARGET          rosbot_1 or rosbot_2
ROSBOT_NS              /rosbot_1 or /rosbot_2
ROSBOT_IMAGE_TOPIC     Compressed camera topic for the selected robot
ROSBOT_PERSON_TOPIC    PersonDetection topic for the selected robot
```

Verify the selected camera:

```bash
ros2 topic info "$ROSBOT_IMAGE_TOPIC"
ros2 topic hz "$ROSBOT_IMAGE_TOPIC"
```

Do not change the variables while processes are running. Stop HAR, Follow, and
Nav2 for the old robot before selecting the new robot, as described in
"Switch robots" below.

## 4. Build the Follow workspace

Enter the inference container:

```bash
docker exec -it jazzy-rosbot-clipgcn bash
```

Build the ROS 2 workspace:

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/person_follow_ws
colcon build --symlink-install
source install/setup.bash
```

Rebuild only after changing the Follow source code or ROS packages.

## 5. Start Nav2, Follow, and HAR

Use separate terminals for Nav2, Follow, and HAR. On the first physical-robot
test, keep the area around the robot clear and have the emergency-stop command
ready.

### Terminal A: Nav2

Enter the navigation container and open its offboard control directory:

```bash
docker exec -it jazzy-rosbot bash
source /opt/ros/jazzy/setup.bash
cd /root/rosbot2-jazzy-image/host/offboard
ROSBOT_TARGET=rosbot_1
```

For robot 2, use:

```bash
ROSBOT_TARGET=rosbot_2
```

Inspect the current runtime and available maps:

```bash
./rosbot-offboard status
./rosbot-offboard topics "$ROSBOT_TARGET"
./rosbot-offboard maps
```

Localize with the existing `lab` map and start Nav2:

```bash
./rosbot-offboard localize "$ROSBOT_TARGET" lab
./rosbot-offboard nav "$ROSBOT_TARGET"
./rosbot-offboard rviz "$ROSBOT_TARGET"
```

To build a live map while navigating instead:

```bash
./rosbot-offboard nav-slam "$ROSBOT_TARGET"
./rosbot-offboard rviz "$ROSBOT_TARGET"
```

Send a navigation goal. Position values are in meters and yaw is in radians:

```bash
./rosbot-offboard goal "$ROSBOT_TARGET" 1.0 0.5 0.0 --yes
```

Stop Nav2 and SLAM for the selected robot:

```bash
./rosbot-offboard stop "$ROSBOT_TARGET"
```

### Terminal B: Follow controller and velocity arbiter

```bash
docker exec -it jazzy-rosbot-clipgcn bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
ros2 launch person_follow_demo person_follow_demo.launch.py \
  namespace:="$ROSBOT_TARGET"
```

Change the selector argument to `rosbot_2` when controlling robot 2. Person
following starts disabled and must be enabled manually after HAR begins
publishing person detections.

### Terminal C: HAR

```bash
docker exec -it jazzy-rosbot-clipgcn bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
cd /workspace/CLIPGCN

python ros_realtime.py \
  --robot-namespace "$ROSBOT_TARGET" \
  --image-transport compressed \
  --config config_final_aug.yaml \
  --class-split-dir data/contrastive_zsl_splits/50_5 \
  --candidate-scope all \
  --unseen-score-scale 1.03 \
  --clipgcn-checkpoint /workspace/CLIPGCN/work_dir/clipgcn_contrastive_50_5/run_20260616_210139/swa_model.pth \
  --pose-source rtmpose \
  --rtmpose-device cuda \
  --temporal-strategy uniform3s \
  --uniform-window-seconds 3 \
  --predict-every 1 \
  --class-config realtime_class_config.csv \
  --top-k 1 \
  --display-filter-window 10 \
  --decision-entropy-threshold 0.30 \
  --decision-temperature 0.05 \
  --cudnn-benchmark \
  --yolo-half \
  --yolo-detect-every 1 \
  --display-width 1280 \
  --display-height 0 \
  --enable-person-follow-output
```

Change the selector argument to `rosbot_2` when controlling robot 2.
`--robot-namespace` automatically selects the matching camera and
PersonDetection topics. Add the following option for headless operation:

```text
--headless
```

### Terminal D: Enable, disable, or emergency-stop Follow

Enter the inference container and select the same robot used by Follow and
HAR:

```bash
docker exec -it jazzy-rosbot-clipgcn bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
```

Enable person following:

```bash
ros2 service call \
  "$ROSBOT_NS/person_follow_controller/set_enabled" \
  std_srvs/srv/SetBool "{data: true}"
```

Disable person following:

```bash
ros2 service call \
  "$ROSBOT_NS/person_follow_controller/set_enabled" \
  std_srvs/srv/SetBool "{data: false}"
```

Activate the emergency stop:

```bash
ros2 topic pub --once \
  "$ROSBOT_NS/follow/emergency_stop" \
  std_msgs/msg/Bool "{data: true}"
```

Release the emergency stop:

```bash
ros2 topic pub --once \
  "$ROSBOT_NS/follow/emergency_stop" \
  std_msgs/msg/Bool "{data: false}"
```

While Nav2 has an active navigation task, the arbiter pauses Follow and forwards
`cmd_vel_nav`. Follow resumes after navigation finishes.

## 6. Switch robots

The following example switches from `rosbot_1` to `rosbot_2`.

1. Disable person following for `rosbot_1`. Activate the emergency stop first
   if necessary.
2. Press `Ctrl+C` in the HAR terminal.
3. Press `Ctrl+C` in the Follow terminal.
4. Stop the old robot's navigation processes in the Nav2 container:

```bash
cd /root/rosbot2-jazzy-image/host/offboard
./rosbot-offboard stop rosbot_1
```

5. If the robots use different Wi-Fi networks, switch the host connection and
   check the `wlp2s0` address and ping again.
6. Select the new target in the Nav2 terminal and restart localization and
   navigation:

```bash
ROSBOT_TARGET=rosbot_2
./rosbot-offboard localize "$ROSBOT_TARGET" lab
./rosbot-offboard nav "$ROSBOT_TARGET"
./rosbot-offboard rviz "$ROSBOT_TARGET"
```

7. Run the selector in each Follow, HAR, and control terminal:

```bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_2
```

Restart Follow and HAR in that order, then enable person following.

## 7. Verification and troubleshooting

Check DDS and the camera connection:

```bash
echo "$ROS_DOMAIN_ID"
echo "$RMW_IMPLEMENTATION"
echo "$CYCLONEDDS_URI"
ros2 topic list | grep -E 'rosbot_[12]'
ros2 topic info -v "$ROSBOT_IMAGE_TOPIC"
```

Inspect Follow data:

```bash
ros2 topic echo "$ROSBOT_NS/follow/person_detection"
ros2 topic echo "$ROSBOT_NS/follow/controller_state"
ros2 topic echo "$ROSBOT_NS/follow/arbiter_state"
```

Inspect velocity publishers:

```bash
ros2 topic info -v "$ROSBOT_NS/cmd_vel"
ros2 topic info -v "$ROSBOT_NS/cmd_vel_nav"
```

The final `cmd_vel` topic should have only `cmd_vel_arbiter` as its publisher.
`cmd_vel_nav` should be published by the Nav2 `controller_server`. Stop any
other node that publishes directly to the final `cmd_vel`, because it may
bypass arbitration and the emergency stop.

Common issues:

- `No image received`: check Wi-Fi, the `wlp2s0` address, DDS peers, Domain ID,
  `ROSBOT_TARGET`, and the camera topic.
- Missing checkpoint or data: these large files are not stored in Git and must
  be restored from local storage.
- Follow does not move: confirm that HAR uses
  `--enable-person-follow-output`, then inspect `person_detection` and the
  Follow enable service.
- Follow is paused: inspect `navigation_active`. This is expected while Nav2
  has an active task.
- No OpenCV or RViz window: rerun `xhost +si:localuser:root` on the host and
  check `DISPLAY`.

## 8. Stop the system

Use this shutdown order:

1. Disable Follow or activate the emergency stop.
2. Press `Ctrl+C` in the HAR and Follow terminals.
3. Run `./rosbot-offboard stop "$ROSBOT_TARGET"` in the Nav2 container.
4. Exit the container shells.
5. Stop the containers if required:

```bash
docker stop jazzy-rosbot-clipgcn jazzy-rosbot
```
