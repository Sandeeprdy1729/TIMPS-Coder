"""
TIMPS-Coder — Step 1: Data Preparation
========================================
Converts your sft_train.jsonl + rlef_train.jsonl into
MLX-LM's expected format (train.jsonl / valid.jsonl).

MLX-LM expects JSONL with a single "text" field per line,
formatted as a chat template:

  {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}

Run: python 1_prepare_data.py
"""

import json
import random
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
SFT_FILE   = "data/raw/sft_train.jsonl"
RLEF_FILE  = "data/raw/rlef_train.jsonl"
OUT_DIR    = "data/processed"
VALID_SPLIT = 0.05       # 5% validation
SEED        = 42
MAX_RLEF_EXAMPLES = 5000  # Use top RLEF pass examples (keeps training fast)
# ────────────────────────────────────────────────────────────

random.seed(SEED)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


def format_chatml(instruction: str, output: str) -> str:
    """Format a single example as ChatML (works with Qwen2.5-Coder)."""
    system = (
        "You are TIMPS-Coder, an expert coding assistant specialized in "
        "debugging and fixing code. Analyze the bug carefully, reason through "
        "the root cause, and provide a clean, correct fix with explanation."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def load_sft(path: str) -> list[dict]:
    """Load SFT data — fields: instruction, output"""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            instruction = d.get("instruction", "").strip()
            output      = d.get("output", "").strip()
            if instruction and output:
                examples.append({"text": format_chatml(instruction, output)})
    print(f"  Loaded {len(examples)} SFT examples from {path}")
    return examples


def load_rlef(path: str, max_examples: int = MAX_RLEF_EXAMPLES) -> list[dict]:
    """
    Load RLEF data — only 'pass' results (execution verified correct).
    Fields: instruction, response, result, reasoning
    We append the reasoning as part of the assistant response for chain-of-thought.
    """
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("result", "").lower() != "pass":
                continue  # Skip failed executions

            instruction = d.get("instruction", "").strip()
            response    = d.get("response", "").strip()
            reasoning   = d.get("reasoning", "").strip()

            if not instruction or not response:
                continue

            # Combine reasoning + code for richer chain-of-thought output
            if reasoning:
                output = f"**Reasoning:**\n{reasoning}\n\n**Fix:**\n{response}"
            else:
                output = response

            examples.append({"text": format_chatml(instruction, output)})

    # Randomly sample to keep dataset balanced
    random.shuffle(examples)
    examples = examples[:max_examples]
    print(f"  Loaded {len(examples)} RLEF 'pass' examples from {path}")
    return examples


def save_split(examples: list[dict], train_path: str, valid_path: str):
    """Split and save train/valid."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - VALID_SPLIT))
    train = examples[:split_idx]
    valid = examples[split_idx:]

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  → Train: {len(train)} | Valid: {len(valid)}")
    return train, valid


def main():
    print("\n📦 TIMPS-Coder Data Preparation")
    print("=" * 45)

    # Load data
    print("\n[1/3] Loading SFT data...")
    sft_examples = load_sft(SFT_FILE)

    print("\n[2/3] Loading RLEF data (pass-only)...")
    rlef_examples = load_rlef(RLEF_FILE)

    # Combine and shuffle
    all_examples = sft_examples + rlef_examples
    random.shuffle(all_examples)
    print(f"\n[3/3] Total examples: {len(all_examples)}")

    # Save
    train_path = os.path.join(OUT_DIR, "train.jsonl")
    valid_path = os.path.join(OUT_DIR, "valid.jsonl")
    save_split(all_examples, train_path, valid_path)

    # Quick sanity check — print one example
    print("\n✅ Sample example (first training row):")
    print("-" * 50)
    with open(train_path) as f:
        sample = json.loads(f.readline())
    print(sample["text"][:500] + "...\n")

    print(f"📁 Output saved to: {OUT_DIR}/")
    print("   → train.jsonl")
    print("   → valid.jsonl")
    print("\nNext: bash 2_train_sft.sh")


if __name__ == "__main__":
    main()
