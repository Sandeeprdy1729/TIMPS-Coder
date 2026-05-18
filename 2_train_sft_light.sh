#!/bin/bash
# ============================================================
# TIMPS-Coder v2 — Light Training (for limited GPU memory)
# ============================================================

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH="./adapters"
DATA_DIR="./data/processed"

echo "========================================================"
echo "  TIMPS-Coder v2 — Light Training"
echo "========================================================"
echo "  Model   : $MODEL"
echo "  Data    : $DATA_DIR"
echo "  Iters   : 800   seq-len: 512 (lighter config)"
echo ""

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "❌  No training data found."
  exit 1
fi

rm -rf "$ADAPTER_PATH"

echo "Phase 1/2 — LoRA fine-tuning (lighter config)..."

python3 -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_PATH" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 1 \
  --grad-accumulation-steps 2 \
  --max-seq-length 512 \
  --iters 800 \
  --learning-rate 2e-6 \
  --val-batches 10 \
  --save-every 400 \
  --steps-per-report 50 \
  --clear-cache-threshold 1

if [ ! -f "$ADAPTER_PATH/adapters.safetensors" ]; then
  echo ""
  echo "❌  Training failed."
  exit 1
fi

echo ""
echo "✅  Phase 1 complete — adapters saved"

# Skip fusing - keep as LoRA for now
echo ""
echo "========================================================"
echo "  ✅  TIMPS-Coder v2 Training Complete!"
echo "========================================================"
echo "  Adapters : $ADAPTER_PATH"
echo "  Next: python3 launch_timps_v2.py"
echo "========================================================"