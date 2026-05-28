#!/usr/bin/env bash
# go_home.py ラッパー — π0-FAST プロジェクト用
# SmolVLA で確認済みのホームポジションに移動する
#
# 使い方:
#   conda activate lerobot-pi0
#   cd ~/sota/projects/pi0-poc
#   bash src/go_home.sh

conda run -n lerobot-pi0 python - <<'EOF'
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import time

HOME_POS = {
    "shoulder_pan.pos":   11.08,
    "shoulder_lift.pos": -103.60,
    "elbow_flex.pos":     96.35,
    "wrist_flex.pos":     65.00,
    "wrist_roll.pos":     -9.80,
    "gripper.pos":         1.00,
}

cfg = SO101FollowerConfig(port="/dev/ttyACM0")
robot = SO101Follower(cfg)
robot.connect()
print("ホームポジションへ移動します...")
robot.send_action(HOME_POS)
time.sleep(2.0)
robot.disconnect()
print("完了:", HOME_POS)
EOF
