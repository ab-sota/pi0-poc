"""
run_robot.py — π0-FAST × SO-101 リアルタイム制御ループ

lerobot-record の --policy.path オプションを使う方法と、
このスクリプトで直接推論する方法の2通りある。
このスクリプトはデバッグ・カスタム制御用。

使い方:
    conda activate lerobot-pi0
    cd ~/sota/projects/pi0-poc

    # 実機テスト（2カメラ）
    python src/run_robot.py \\
        --checkpoint models/pi0fast_so101_v1/checkpoints/050000/pretrained_model \\
        --task "pick up the blue pen and lift it up" \\
        --camera_id 0 --wrist_camera_id 4 \\
        --robot_port /dev/ttyACM0

    # ドライラン（サーボ送信なし）
    python src/run_robot.py --checkpoint ... --dry_run

    # lerobot-record 経由（推奨、評価エピソード記録も同時）:
    lerobot-record \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=follower_arm \\
        --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \\
        --dataset.repo_id=ShunAB/eval_pi0fast_so101 \\
        --dataset.single_task="pick up the blue pen and lift it up" \\
        --dataset.num_episodes=10 \\
        --policy.path=models/pi0fast_so101_v1/checkpoints/050000/pretrained_model \\
        --display_data=true
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# SO-101 関節キー (LeRobot 最新版)
_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="π0-FAST × SO-101 制御ループ")
    p.add_argument("--checkpoint", required=True, help="pretrained_model ディレクトリ")
    p.add_argument("--task", default="pick up the blue pen and lift it up")
    p.add_argument("--camera_id", type=int, default=0, help="top カメラ OpenCV ID")
    p.add_argument("--wrist_camera_id", type=int, default=None, help="wrist カメラ OpenCV ID")
    p.add_argument("--robot_port", default="/dev/ttyACM0")
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--control_freq", type=float, default=30.0, help="制御周波数 Hz")
    p.add_argument("--delta_limit", type=float, default=5.0, help="最大関節変化量 deg")
    p.add_argument("--img_size", type=int, default=224, help="pi0-FAST は 224px 推奨")
    p.add_argument("--dry_run", action="store_true", help="サーボ送信なし")
    p.add_argument("--record_video", action="store_true", help="動画を results/ に保存")
    return p.parse_args()


def capture_frame(cap: cv2.VideoCapture, img_size: int) -> np.ndarray:
    """カメラから1フレーム取得して (C, H, W) float32 [0,1] に変換"""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("カメラ読み取り失敗")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
    return frame_resized.astype(np.float32) / 255.0


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"デバイス: {device}")

    # ── モデルロード ─────────────────────────────────────────────────────────
    logger.info(f"π0-FAST チェックポイント読み込み: {args.checkpoint}")
    from lerobot.policies.pi0_fast import PI0FastPolicy

    policy = PI0FastPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(device).eval()
    logger.info("モデルロード完了")

    # ── カメラ初期化 ─────────────────────────────────────────────────────────
    cap_top = cv2.VideoCapture(args.camera_id)
    cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_top.set(cv2.CAP_PROP_FPS, 30)
    if not cap_top.isOpened():
        logger.error(f"top カメラ (id={args.camera_id}) を開けません")
        sys.exit(1)

    cap_wrist = None
    if args.wrist_camera_id is not None:
        cap_wrist = cv2.VideoCapture(args.wrist_camera_id)
        cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_wrist.set(cv2.CAP_PROP_FPS, 30)
        if not cap_wrist.isOpened():
            logger.error(f"wrist カメラ (id={args.wrist_camera_id}) を開けません")
            sys.exit(1)

    logger.info(f"カメラ初期化完了: top={args.camera_id}" +
                (f", wrist={args.wrist_camera_id}" if cap_wrist else ""))

    # ── ロボット初期化 ────────────────────────────────────────────────────────
    robot = None
    if not args.dry_run:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        cfg = SO101FollowerConfig(port=args.robot_port)
        robot = SO101Follower(cfg)
        robot.connect()
        logger.info(f"SO-101 接続完了: {args.robot_port}")

    # ── 動画記録 ──────────────────────────────────────────────────────────────
    video_writer = None
    if args.record_video:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = results_dir / f"run_pi0fast_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.control_freq,
                                       (640 * (2 if cap_wrist else 1), 480))
        logger.info(f"動画記録: {video_path}")

    # ── タスクのトークン化（一度だけ） ────────────────────────────────────────
    tokenizer = policy._paligemma_tokenizer
    task_enc = tokenizer(args.task, return_tensors="pt", padding=True)
    task_tokens = task_enc["input_ids"].to(device)
    task_mask   = task_enc["attention_mask"].bool().to(device)
    logger.info(f"タスクトークン: shape={task_tokens.shape}")

    # ── 制御ループ ─────────────────────────────────────────────────────────────
    logger.info(f"制御開始: task='{args.task}', max_steps={args.max_steps}")
    dt = 1.0 / args.control_freq
    prev_joints = np.zeros(6, dtype=np.float32)

    try:
        for step in range(args.max_steps if args.max_steps > 0 else int(1e9)):
            t_start = time.perf_counter()

            # 画像取得
            img_top = capture_frame(cap_top, args.img_size)
            imgs = {"observation.images.base_0_rgb": torch.from_numpy(img_top).permute(2, 0, 1)}
            if cap_wrist:
                img_wrist = capture_frame(cap_wrist, args.img_size)
                imgs["observation.images.left_wrist_0_rgb"] = torch.from_numpy(img_wrist).permute(2, 0, 1)

            # 関節状態取得
            if robot is not None:
                obs = robot.get_observation()
                joints_deg = np.array([obs[k] for k in _JOINT_KEYS], dtype=np.float32)
            else:
                joints_deg = prev_joints.copy()

            # バッチ構築
            batch = {k: v.unsqueeze(0).to(device) for k, v in imgs.items()}
            batch["observation.state"] = torch.from_numpy(joints_deg).unsqueeze(0).to(device)
            batch["observation.language.tokens"]       = task_tokens
            batch["observation.language.attention_mask"] = task_mask

            # 推論
            with torch.inference_mode():
                action = policy.select_action(batch)

            # action shape: (1, action_dim) or (1, chunk, action_dim)
            if action.dim() == 3:
                action = action[0, 0]   # 最初のステップのみ使用
            else:
                action = action[0]

            action_np = action.cpu().numpy().astype(np.float64)

            # delta clamp
            delta = action_np - joints_deg
            delta = np.clip(delta, -args.delta_limit, args.delta_limit)
            target = joints_deg + delta

            # サーボ送信
            if robot is not None:
                cmd = {k: float(target[i]) for i, k in enumerate(_JOINT_KEYS)}
                robot.send_action(cmd)

            prev_joints = target.astype(np.float32)

            # 動画
            if video_writer is not None:
                ret_top, frame_top = cap_top.read()
                if ret_top:
                    if cap_wrist:
                        ret_w, frame_w = cap_wrist.read()
                        frame_combined = cv2.hconcat([frame_top, frame_w if ret_w else frame_top])
                    else:
                        frame_combined = frame_top
                    video_writer.write(frame_combined)

            if step % 50 == 0:
                logger.info(f"step={step:4d}  joints={np.round(target, 1)}")

            # 周波数制御
            elapsed = time.perf_counter() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("ユーザー中断")
    finally:
        cap_top.release()
        if cap_wrist:
            cap_wrist.release()
        if video_writer:
            video_writer.release()
        if robot:
            robot.disconnect()
        logger.info("終了")


if __name__ == "__main__":
    main()
