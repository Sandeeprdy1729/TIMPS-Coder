#!/bin/bash
# ============================================================
# TIMPS-Coder Path B — Retrain with Clean Dataset
# Mac M2 Air 8GB optimized
# ============================================================

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH="./adapters"
DATA_DIR="./data/processed"

echo "🚀 TIMPS-Coder Retraining — Clean Dataset"
echo "   Model:   $MODEL"
echo "   Data:    $DATA_DIR"
echo "   Train:   $(wc -l < $DATA_DIR/train.jsonl) examples"
echo ""

# Wipe old run
rm -rf "$ADAPTER_PATH" ./timps-coder-fused

# ── Fine-tune ────────────────────────────────────────────────
python3 -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_PATH" \
  --batch-size 2 \
  --grad-accumulation-steps 4 \
  --max-seq-length 1024 \
  --iters 3000 \
  --learning-rate 5e-6 \
  --val-batches 20 \
  --save-every 500 \
  --steps-per-report 100 \
  --grad-checkpoint

if [ ! -f "$ADAPTER_PATH/adapters.safetensors" ]; then
  echo "❌ Training failed."
  exit 1
fi

# ── Fuse ─────────────────────────────────────────────────────
echo ""
echo "🔗 Fusing adapters..."
python3 -m mlx_lm fuse \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --save-path "./timps-coder-fused"

if [ ! -f "./timps-coder-fused/config.json" ]; then
  echo "❌ Fuse failed."
  exit 1
fi

echo ""
echo "✅ Done! Test:"
echo "   python3 test.py"