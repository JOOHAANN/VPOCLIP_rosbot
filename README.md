# CLIPGCN ROSbot 运行指南

本项目在 ROS 2 Jazzy 下读取 ROSbot 摄像头，运行 CLIPGCN 动作识别（HAR），
并把同一次 YOLO 检测得到的人体目标发送给人物跟随控制器。Nav2 和人物跟随
共用速度仲裁器，控制优先级为：

```text
急停 > Nav2 导航 > 人物跟随 > 停车
```

数据流：

```text
ROSbot 摄像头 -> CLIPGCN / YOLO -> HAR 结果与 PersonDetection
                                         |
                                         v
                                 person_follow_controller
                                         |
                                         v
Nav2 -> cmd_vel_nav ---------> cmd_vel_arbiter -> ROSbot cmd_vel
人物跟随 -> cmd_vel_follow ---/
```

下面的命令按当前工作站目录和容器配置编写：

```text
/home/youhan/HAR/CLIPGCN             HAR 源码、配置和本地模型
/home/youhan/HAR/VPOCLIP_rosbot      Docker Compose 配置
/home/youhan/HAR/person_follow_ws    Follow ROS 2 工作区
/home/youhan/HAR/X3D                 X3D 代码
/home/youhan/HAR/CTR-GCN_17          CTR-GCN 代码
/home/youhan/HAR/yolov5              YOLOv5 代码
```

模型、数据集和训练输出已被 `.gitignore` 排除，不会上传到 GitHub。运行前必须
在本机恢复以下资源：

```text
data/contrastive_zsl_splits/50_5/
work_dir/clipgcn_contrastive_50_5/run_20260616_210139/swa_model.pth
local_models/
```

## 1. 配置和检查 Wi-Fi

当前 DDS 配置使用无线接口 `wlp2s0`，ROS Domain ID 为 `10`，已配置的 peer
地址为 `10.186.13.100`、`10.186.13.101` 和 `10.186.13.30`。

查看无线接口和可用网络：

```bash
nmcli device status
nmcli device wifi list ifname wlp2s0
```

连接已经保存过的 ROSbot Wi-Fi：

```bash
nmcli connection show
nmcli connection up id "你的ROSbot Wi-Fi连接名"
```

首次连接：

```bash
ROSBOT_WIFI_SSID="你的Wi-Fi名称"
ROSBOT_WIFI_PASSWORD="你的Wi-Fi密码"
nmcli device wifi connect "$ROSBOT_WIFI_SSID" \
  password "$ROSBOT_WIFI_PASSWORD" \
  ifname wlp2s0
unset ROSBOT_WIFI_PASSWORD
```

检查本机地址和机器人连通性：

```bash
ip -4 -br address show wlp2s0
ping -c 1 10.186.13.100
ping -c 1 10.186.13.101
```

如果无线接口不叫 `wlp2s0`，先在
`/home/youhan/HAR/VPOCLIP_rosbot/compose.yaml` 中修改 CycloneDDS 的
`NetworkInterface name`，并同步修改 `jazzy-rosbot` 导航容器使用的 DDS
接口。修改 Compose 后需要重新创建推理容器。

Wi-Fi 密码只应在终端或本机 NetworkManager 中配置，不要写进本仓库。

## 2. 启动容器

允许容器显示 OpenCV/RViz 窗口，然后启动两个已有容器：

```bash
xhost +si:localuser:root
docker start jazzy-rosbot jazzy-rosbot-clipgcn
docker ps --filter name=jazzy-rosbot
```

首次创建推理容器，或修改了 Dockerfile、requirements/Compose 配置时：

```bash
cd /home/youhan/HAR/VPOCLIP_rosbot
docker compose up -d --build
```

平时不要使用 `docker compose down`，因为它会删除推理容器。正常停止使用：

```bash
docker stop jazzy-rosbot-clipgcn jazzy-rosbot
```

## 3. 选择 rosbot_1 或 rosbot_2

Follow 和 HAR 的每一个新终端都必须执行一次机器人选择脚本。选择机器人 1：

```bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
```

选择机器人 2：

```bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_2
```

脚本会设置：

```text
ROSBOT_TARGET          rosbot_1 或 rosbot_2
ROSBOT_NS              /rosbot_1 或 /rosbot_2
ROSBOT_IMAGE_TOPIC     对应机器人的压缩图像话题
ROSBOT_PERSON_TOPIC    对应机器人的 PersonDetection 话题
```

检查所选机器人的摄像头：

```bash
ros2 topic info "$ROSBOT_IMAGE_TOPIC"
ros2 topic hz "$ROSBOT_IMAGE_TOPIC"
```

不要在进程运行中直接切换变量。切换机器人时，先按本文“切换机器人”一节停止
旧机器人的 HAR、Follow 和 Nav2，再在各终端重新执行选择脚本。

## 4. 首次构建 Follow 工作区

进入推理容器：

```bash
docker exec -it jazzy-rosbot-clipgcn bash
```

构建：

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/person_follow_ws
colcon build --symlink-install
source install/setup.bash
```

只有代码或 ROS 包发生变化时才需要重新构建。

## 5. 启动顺序

建议分别打开三个终端：Nav、Follow、HAR。第一次真机运行时应保持机器人周围
无障碍，并准备好急停命令。

### 终端 A：Nav2

进入导航容器并设置控制脚本路径：

```bash
docker exec -it jazzy-rosbot bash
source /opt/ros/jazzy/setup.bash
cd /root/rosbot2-jazzy-image/host/offboard
ROSBOT_TARGET=rosbot_1
```

如需机器人 2，把最后一行改为：

```bash
ROSBOT_TARGET=rosbot_2
```

查看状态和地图：

```bash
./rosbot-offboard status
./rosbot-offboard topics "$ROSBOT_TARGET"
./rosbot-offboard maps
```

使用已有的 `lab` 地图：

```bash
./rosbot-offboard localize "$ROSBOT_TARGET" lab
./rosbot-offboard nav "$ROSBOT_TARGET"
./rosbot-offboard rviz "$ROSBOT_TARGET"
```

如果要实时建图并导航：

```bash
./rosbot-offboard nav-slam "$ROSBOT_TARGET"
./rosbot-offboard rviz "$ROSBOT_TARGET"
```

发送导航目标，单位为米和弧度：

```bash
./rosbot-offboard goal "$ROSBOT_TARGET" 1.0 0.5 0.0 --yes
```

停止该机器人的 Nav2/SLAM：

```bash
./rosbot-offboard stop "$ROSBOT_TARGET"
```

### 终端 B：Follow 控制器和速度仲裁器

```bash
docker exec -it jazzy-rosbot-clipgcn bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
ros2 launch person_follow_demo person_follow_demo.launch.py \
  namespace:="$ROSBOT_TARGET"
```

使用机器人 2 时，把选择脚本的参数改成 `rosbot_2`。Follow 启动时默认关闭，
必须在 HAR 已经发布人体检测后手动开启。

### 终端 C：HAR

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

使用机器人 2 时，把选择脚本的参数改成 `rosbot_2`。`--robot-namespace`
会自动选择对应的摄像头和 PersonDetection 话题。无显示器运行时在命令末尾加：

```text
--headless
```

### 终端 D：开启、关闭和急停 Follow

进入推理容器并选择同一个机器人：

```bash
docker exec -it jazzy-rosbot-clipgcn bash
source /opt/ros/jazzy/setup.bash
source /workspace/person_follow_ws/install/setup.bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_1
```

开启人物跟随：

```bash
ros2 service call \
  "$ROSBOT_NS/person_follow_controller/set_enabled" \
  std_srvs/srv/SetBool "{data: true}"
```

关闭人物跟随：

```bash
ros2 service call \
  "$ROSBOT_NS/person_follow_controller/set_enabled" \
  std_srvs/srv/SetBool "{data: false}"
```

急停：

```bash
ros2 topic pub --once \
  "$ROSBOT_NS/follow/emergency_stop" \
  std_msgs/msg/Bool "{data: true}"
```

解除急停：

```bash
ros2 topic pub --once \
  "$ROSBOT_NS/follow/emergency_stop" \
  std_msgs/msg/Bool "{data: false}"
```

Nav2 执行导航任务时，仲裁器会自动暂停 Follow 并优先转发
`cmd_vel_nav`；导航结束后再恢复 Follow。

## 6. 切换机器人

以下示例从 `rosbot_1` 切换到 `rosbot_2`。

1. 在终端 D 关闭 `rosbot_1` 的人物跟随；需要时先发送急停。
2. 在 HAR 终端按 `Ctrl+C`。
3. 在 Follow 终端按 `Ctrl+C`。
4. 在 Nav 容器中停止旧机器人的进程：

```bash
cd /root/rosbot2-jazzy-image/host/offboard
./rosbot-offboard stop rosbot_1
```

5. 如两个机器人使用不同 Wi-Fi，先在宿主机切换连接并重新检查
   `wlp2s0` 地址和 ping。
6. 在 Nav 终端设置新目标并重新启动 localization/Nav2：

```bash
ROSBOT_TARGET=rosbot_2
./rosbot-offboard localize "$ROSBOT_TARGET" lab
./rosbot-offboard nav "$ROSBOT_TARGET"
./rosbot-offboard rviz "$ROSBOT_TARGET"
```

7. 在 Follow、HAR、控制终端分别重新执行：

```bash
source /workspace/person_follow_ws/select_rosbot.sh rosbot_2
```

然后按照启动顺序重新运行 Follow、HAR，最后再开启人物跟随。

## 7. 检查与故障排除

检查 DDS 和摄像头：

```bash
echo "$ROS_DOMAIN_ID"
echo "$RMW_IMPLEMENTATION"
echo "$CYCLONEDDS_URI"
ros2 topic list | grep -E 'rosbot_[12]'
ros2 topic info -v "$ROSBOT_IMAGE_TOPIC"
```

检查 Follow 数据：

```bash
ros2 topic echo "$ROSBOT_NS/follow/person_detection"
ros2 topic echo "$ROSBOT_NS/follow/controller_state"
ros2 topic echo "$ROSBOT_NS/follow/arbiter_state"
```

检查速度发布者：

```bash
ros2 topic info -v "$ROSBOT_NS/cmd_vel"
ros2 topic info -v "$ROSBOT_NS/cmd_vel_nav"
```

最终 `cmd_vel` 应只有 `cmd_vel_arbiter` 一个 publisher；`cmd_vel_nav` 应由
Nav2 `controller_server` 发布。若有其他节点直接发布 `cmd_vel`，先停止它，
否则可能绕过仲裁和急停。

常见问题：

- `No image received`：检查 Wi-Fi、`wlp2s0` 地址、DDS peer、Domain ID、
  `ROSBOT_TARGET` 和摄像头话题。
- 找不到 checkpoint/data：这些大文件不会进入 Git，需从本地备份恢复。
- Follow 不动：确认 HAR 使用了 `--enable-person-follow-output`，再检查
  `person_detection` 和 Follow enable 服务。
- Follow 被暂停：检查 `navigation_active`；Nav2 有活动任务时这是正常行为。
- OpenCV/RViz 无窗口：在宿主机重新运行 `xhost +si:localuser:root`，并检查
  `DISPLAY`。

## 8. 停止系统

推荐顺序：

1. 关闭 Follow 或发送急停。
2. 在 HAR 和 Follow 终端按 `Ctrl+C`。
3. 在 Nav 容器运行 `./rosbot-offboard stop "$ROSBOT_TARGET"`。
4. 退出各容器 shell。
5. 需要时停止容器：

```bash
docker stop jazzy-rosbot-clipgcn jazzy-rosbot
```
