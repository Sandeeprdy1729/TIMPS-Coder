#!/bin/bash
# ============================================================
# TIMPS-Coder v2 — SFT Training Pipeline (Adapter-based)
# Mac M1 / M2 / M3  (8 GB RAM optimised)
#
# Training on HuggingFace public datasets:
#   - SWE-bench/SWE-bench_Verified    (GitHub issue → patch)
#   - TIGER-Lab/SWE-Next-SFT-Trajectories (agentic edit traces)
#   - newfacade/LeetCodeDataset        (algorithm problems)
#   - WaltonFuture/agentic-sft-new     (tool use + bash)
#   - DeepNLP/Coding-Agent-Github-2025-Feb (coding agent traces)
#
# Uses LoRA adapters (no fusion) - lighter and more flexible
# ============================================================

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"

ADAPTER_PATH="./adapters"
DATA_DIR="./data/processed"

echo "========================================================"
echo "  TIMPS-Coder v2 — Training"
echo "========================================================"
echo "  Model   : $MODEL"
echo "  Data    : $DATA_DIR"
echo "  LoRA    : $ADAPTER_PATH  (rank=16)"
echo "  Iters   : 2000   seq-len: 2048"
echo ""

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "❌  No training data found.  Run first:"
  echo "      python3 1_prepare_data_v2.py"
  exit 1
fi

NTRAIN=$(wc -l < "$DATA_DIR/train.jsonl")
NVALID=$(wc -l < "$DATA_DIR/valid.jsonl")
echo "  Train examples : $NTRAIN"
echo "  Valid examples : $NVALID"
echo ""

# Clean previous adapters
rm -rf "$ADAPTER_PATH"
mkdir -p "$ADAPTER_PATH"

# ── LoRA Fine-tuning ─────────────────────────────────
echo "LoRA Fine-tuning..."
echo ""

python3 -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_PATH" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --max-seq-length 2048 \
  --iters 2000 \
  --learning-rate 5e-6 \
  --val-batches 20 \
  --save-every 500 \
  --steps-per-report 100 \
  --grad-checkpoint \
  --clear-cache-threshold 1

if [ ! -f "$ADAPTER_PATH/adapters.safetensors" ]; then
  echo ""
  echo "❌  Training failed: adapters not saved."
  exit 1
fi

echo ""
echo "========================================================"
echo "  ✅  TIMPS-Coder v2 Training Complete!"
echo "========================================================"
echo "  LoRA adapters : $ADAPTER_PATH"
echo ""
echo "  To use the model:"
echo "    python3 launch_timps_v2.py       # showcase demo"
echo "    python3 3_benchmark_v2.py        # full benchmark"
echo "    python3 publish.py               # push to HuggingFace"
echo "========================================================"