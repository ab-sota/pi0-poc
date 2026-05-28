# π0-FAST × SO-101 PoC

Physical Intelligence の π0-FAST を SO-101 ロボットアームに適用する検証プロジェクト。

## 概要

| 項目 | 内容 |
|------|------|
| ベースモデル | `lerobot/pi0fast-base` (4B params) |
| アーキテクチャ | PaliGemma + Expert Gemma + FAST tokenizer (DCT) |
| データセット | `ShunAB/so101_pick_and_place_v2` (80ep, top+wrist 2カメラ) |
| conda 環境 | `lerobot-pi0` |
| タスク | "pick up the blue pen and lift it up" |

## セットアップ

```bash
# 環境は作成済み
conda activate lerobot-pi0
cd ~/sota/projects/pi0-poc
```

## ファインチューニング

```bash
conda activate lerobot-pi0
cd ~/sota/projects/pi0-poc

# バックグラウンド実行
nohup bash src/finetune.sh > /tmp/pi0fast_finetune.log 2>&1 &
disown $!
tail -f /tmp/pi0fast_finetune.log
```

学習パラメータ:
- steps: 50,000
- batch_size: 8
- dtype: bfloat16 + gradient_checkpointing
- 推定学習時間: ~3–5時間 (RTX 5090)

## 実機テスト

```bash
conda activate lerobot-pi0
cd ~/sota/projects/pi0-poc

# ①ホームポジション
bash src/go_home.sh

# ②実機テスト（run_robot.py）
python src/run_robot.py \
    --checkpoint models/pi0fast_so101_v1/checkpoints/050000/pretrained_model \
    --task "pick up the blue pen and lift it up" \
    --camera_id 0 --wrist_camera_id 4 \
    --robot_port /dev/ttyACM0 \
    --max_steps 1500 --delta_limit 5.0 --record_video

# ③lerobot-record 経由（成功率の定量評価に推奨）
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=ShunAB/eval_pi0fast_so101 \
    --dataset.single_task="pick up the blue pen and lift it up" \
    --dataset.num_episodes=10 \
    --policy.path=models/pi0fast_so101_v1/checkpoints/050000/pretrained_model \
    --display_data=true
```

## ハードウェア構成

| 機器 | 設定 |
|------|------|
| GPU | RTX 5090 Laptop (25.1GB VRAM) |
| カメラ top | /dev/video0, camera_id=0 |
| カメラ wrist | /dev/video4, camera_id=4 |
| SO-101 follower | /dev/ttyACM0 |
| SO-101 leader | /dev/ttyACM1 |

## 結果

| バージョン | データ | ステップ | 結果 |
|-----------|--------|---------|------|
| pi0fast_so101_v1 | 80ep, 2cam | 50k | （実施予定） |

## 参照

- [SmolVLA PoC](../smolvla-poc/) — 先行プロジェクト（v2 成功済み）
- [HF: lerobot/pi0fast-base](https://huggingface.co/lerobot/pi0fast-base)
- [Physical Intelligence openpi (JAX)](https://github.com/Physical-Intelligence/openpi)
