#!/usr/bin/env bash
# finetune.sh — π0-FAST を SO-101 データセットでファインチューニング
#
# 前提:
#   - conda 環境 lerobot-pi0 がアクティブ
#   - データセット: ShunAB/so101_pick_and_place_v2 (HF Hub, 80ep, top+wrist 2カメラ)
#   - GPU: RTX 5090 Laptop (25.1GB VRAM)
#
# 使い方:
#   conda activate lerobot-pi0
#   cd ~/sota/projects/pi0-poc
#   bash src/finetune.sh
#
# バックグラウンド実行:
#   nohup bash src/finetune.sh > /tmp/pi0fast_finetune.log 2>&1 &
#   disown $!
#   tail -f /tmp/pi0fast_finetune.log

set -e

# ── パス設定 ──────────────────────────────────────────────────────────────────
DATASET_ID="ShunAB/so101_pick_and_place_v2"
BASE_MODEL="lerobot/pi0fast-base"
OUTPUT_DIR="${HOME}/sota/projects/pi0-poc/models/pi0fast_so101_v1"
LOG_FILE="/tmp/pi0fast_finetune.log"

# ── 学習パラメータ ─────────────────────────────────────────────────────────────
# π0-FAST は SmolVLA の5倍高速なので同じ2.5時間なら ~150k step 相当だが
# まず 50k step で品質を確認する
STEPS=50000
BATCH_SIZE=8      # 4B bfloat16 + gradient_checkpointing → ~20GB
SAVE_FREQ=10000
SEED=42

# ── 実行 ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "π0-FAST Fine-tuning v1 (SO-101, 2-camera)"
echo "  dataset   : ${DATASET_ID}"
echo "  base model: ${BASE_MODEL}"
echo "  output    : ${OUTPUT_DIR}"
echo "  steps     : ${STEPS}  batch: ${BATCH_SIZE}  save_freq: ${SAVE_FREQ}"
echo "=========================================="
echo "開始時刻: $(date)"

lerobot-train \
    --dataset.repo_id="${DATASET_ID}" \
    --policy.type=pi0_fast \
    --policy.pretrained_name_or_path="${BASE_MODEL}" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.gradient_checkpointing=true \
    --output_dir="${OUTPUT_DIR}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --seed="${SEED}"

echo "完了時刻: $(date)"
echo "チェックポイント: ${OUTPUT_DIR}/checkpoints/"
