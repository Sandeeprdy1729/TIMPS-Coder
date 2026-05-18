#!/bin/bash
# ============================================================
# TIMPS-Coder Fine-tuning Setup — Mac M2 Optimized
# Run this ONCE before anything else
# ============================================================

echo "🔧 Setting up TIMPS-Coder training environment..."

# 1. Install MLX and MLX-LM (Apple Silicon native — uses Neural Engine + GPU)
pip install mlx mlx-lm

# 2. HuggingFace tools for publishing
pip install huggingface_hub transformers

# 3. Data tools
pip install datasets tqdm

# 4. Optional: evaluation
pip install evaluate

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. python 1_prepare_data.py"
echo "  2. bash 2_train_sft.sh"
echo "  3. python 3_publish.py"
