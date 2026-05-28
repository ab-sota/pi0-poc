# π0-FAST × SO-101 PoC — 作業進捗ログ

プロジェクト: `~/sota/projects/pi0-poc/`
conda 環境: `lerobot-pi0` (Python 3.12, lerobot 0.5.2)
GitHub: `git@github.com:ab-sota/pi0-poc.git`

---

## フェーズ全体の状態

| Phase | 内容 | 状態 |
|-------|------|------|
| 環境構築 | conda `lerobot-pi0` (lerobot 0.5.2) | ✅ 完了 |
| モデルロード確認 | `lerobot/pi0fast-base` (3.45B, bfloat16) | ✅ 完了 |
| PaliGemma 利用許諾 | `google/paligemma-3b-pt-224` への同意 | ✅ 完了 |
| ゼロショット調査 | state 32次元不一致のため不可と確認 | ✅ 確認済み |
| ファインチューニング準備 | `finetune.sh` / `run_robot.py` 作成 | ✅ 完了 |
| **ファインチューニング実行** | **50k step** | ⏳ 未実施 |
| 実機テスト | — | ⏳ 未実施 |

---

## 2026-05-28 夜 — 環境構築・ゼロショット調査

### 環境構築

```bash
conda create -n lerobot-pi0 python=3.12 -y
conda activate lerobot-pi0
pip install "lerobot[dataset,pi0]@git+https://github.com/huggingface/lerobot.git"
pip install datasets transformers accelerate
```

インストール後の確認:
```
lerobot 0.5.2
torch 2.11.0
PI0FastPolicy import OK
lerobot/pi0fast-base ロード成功: 3.45B params, dtype=torch.bfloat16
```

### ゼロショット不可を確認

`pi0fast-base` のベースモデル設定:

| 項目 | ベースモデル期待値 | SO-101 実際 |
|------|-----------------|-------------|
| `observation.state` 次元 | **32** | 6 |
| `action` 次元 | **32** | 6 |
| カメラキー | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` | `top`, `wrist` |

→ ゼロショットでは全 `<bos>` を出力（意味のある推論不可）  
→ **ファインチューニング必須**（学習時にデータセットの次元・キーに合わせてくれる）

### PaliGemma 利用許諾

`lerobot/pi0fast-base` ロードに `google/paligemma-3b-pt-224` の利用許諾が必要。
以下で同意済み: https://huggingface.co/google/paligemma-3b-pt-224

---

## ファインチューニング手順

```bash
conda activate lerobot-pi0
cd ~/sota/projects/pi0-poc

# バックグラウンド実行
nohup bash src/finetune.sh > /tmp/pi0fast_finetune.log 2>&1 &
disown $!
tail -f /tmp/pi0fast_finetune.log
```

設定 (`src/finetune.sh`):

| 項目 | 値 |
|------|-----|
| ベースモデル | `lerobot/pi0fast-base` |
| データセット | `ShunAB/so101_pick_and_place_v2` (80ep, top+wrist 2カメラ) |
| steps | 50,000 |
| batch_size | 8 |
| dtype | bfloat16 + gradient_checkpointing |
| 推定学習時間 | 3–5時間 (RTX 5090) |
| 保存先 | `models/pi0fast_so101_v1/` |

---

## 実機テスト手順（ファインチューニング完了後）

```bash
conda activate lerobot-pi0
cd ~/sota/projects/pi0-poc

# ①ホームポジション
bash src/go_home.sh

# ②実機テスト
python src/run_robot.py \
    --checkpoint models/pi0fast_so101_v1/checkpoints/050000/pretrained_model \
    --task "pick up the blue pen and lift it up" \
    --camera_id 0 --wrist_camera_id 4 \
    --robot_port /dev/ttyACM0 \
    --max_steps 1500 --delta_limit 5.0 --record_video

# ③定量評価（lerobot-record 経由）
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                      wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=ShunAB/eval_pi0fast_so101 \
    --dataset.single_task="pick up the blue pen and lift it up" \
    --dataset.num_episodes=10 \
    --policy.path=models/pi0fast_so101_v1/checkpoints/050000/pretrained_model \
    --display_data=true
```

---

## 結果ログ

| バージョン | データ | ステップ | 結果 |
|-----------|--------|---------|------|
| pi0fast_so101_v1 | 80ep, top+wrist | 50k | ⏳ 未実施 |

---

## SmolVLA との比較予定

| 項目 | SmolVLA v2 | π0-FAST v1 |
|------|-----------|-----------|
| パラメータ数 | ~500M | 3.45B |
| 学習時間 | 2.5h (30k step) | 推定 3-5h (50k step) |
| v2 実機テスト | ✅ **成功** | ⏳ 実施予定 |
| 成功率 | 未測定 | 未測定 |

## 参照

- [SmolVLA 進捗](progress_smolvla.md)
- [HF: lerobot/pi0fast-base](https://huggingface.co/lerobot/pi0fast-base)
- [Physical Intelligence openpi (JAX)](https://github.com/Physical-Intelligence/openpi)
