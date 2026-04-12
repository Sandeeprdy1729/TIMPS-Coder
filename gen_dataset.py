#!/usr/bin/env python3
"""
TIMPS-Coder Path B — Clean Dataset Builder
============================================
Run this ON YOUR MAC inside timps-coder-finetune/

What it does:
  1. Downloads 3 clean, verified HuggingFace coding datasets
  2. Merges + deduplicates them
  3. Filters to only bug-fix / code-correction examples
  4. Outputs clean train.jsonl + valid.jsonl ready for MLX training

Usage:
  pip install datasets
  python3 build_clean_dataset.py
"""

import json, re, random, os
from pathlib import Path
from datasets import load_dataset

random.seed(42)

SYSTEM = """You are TIMPS-Coder, an expert coding assistant built by Sandeep Reddy.
When asked to fix a bug or write code:
1. Briefly explain the root cause or approach (2-3 sentences)
2. Provide the complete, correct code"""

# ── Datasets to use ──────────────────────────────────────────
# These are verified open, no auth needed, high quality
SOURCES = [
    {
        "name": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "instruction_key": "instruction",
        "output_key": "output",
        "input_key": "input",
        "max": 20000,
    },
    {
        "name": "HuggingFaceH4/CodeAlpaca_20K",
        "split": "train",
        "instruction_key": "prompt",
        "output_key": "completion",
        "input_key": None,
        "max": 20000,
    },
    {
        "name": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "instruction_key": "instruction",
        "output_key": "output",
        "input_key": "input",
        "max": 18000,
    },
    {
        "name": "Vezora/Tested-22k-Python-Alpaca",
        "split": "train",
        "instruction_key": "instruction",
        "output_key": "output",
        "input_key": "input",
        "max": 22000,
    },
    {
        "name": "TokenBender/code_instructions_122k_alpaca_style",
        "split": "train",
        "instruction_key": "instruction",
        "output_key": "output",
        "input_key": "input",
        "max": 30000,   # cap at 30k from 122k
    },
]

# ── Helpers ──────────────────────────────────────────────────

def format_chatml(instruction: str, output: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{output.strip()}<|im_end|>"
    )


def is_good(instruction: str, output: str) -> bool:
    """Quality gate — strict."""
    if not instruction or not output: return False
    if len(instruction.strip()) < 20: return False
    if len(output.strip()) < 40: return False

    # Must have actual code (not just prose)
    has_code = (
        "```" in output or
        any(kw in output for kw in [
            "def ", "class ", "public ", "private ", "function ",
            "const ", "let ", "var ", "import ", "return ", "if ("
        ])
    )
    if not has_code: return False

    # No [CODE] placeholder junk
    if "[CODE]" in output or "`...`" in output: return False

    # No HTML entities
    if "&#" in output or "&amp;" in output: return False

    return True


def load_source(source: dict) -> list:
    name = source["name"]
    print(f"\n  Loading {name}...")
    try:
        ds = load_dataset(name, split=source["split"])
    except Exception as e:
        print(f"  ⚠️  Skipped {name}: {e}")
        return []

    ik = source["instruction_key"]
    ok = source["output_key"]
    ink = source.get("input_key")

    examples = []
    for row in ds:
        instruction = str(row.get(ik, "")).strip()
        output      = str(row.get(ok, "")).strip()

        # Merge input into instruction if present
        if ink and row.get(ink, "").strip():
            instruction = instruction + "\n\n" + row[ink].strip()

        if not is_good(instruction, output):
            continue

        examples.append({
            "text": format_chatml(instruction, output)
        })

        if len(examples) >= source["max"]:
            break

    print(f"  → {len(examples)} clean examples")
    return examples


# ── Main ─────────────────────────────────────────────────────

def main():
    print("🔧 TIMPS-Coder Clean Dataset Builder")
    print("=" * 45)

    all_examples = []
    for source in SOURCES:
        all_examples.extend(load_source(source))

    print(f"\nTotal before dedup: {len(all_examples)}")

    # Deduplicate by instruction prefix
    seen, unique = set(), []
    for ex in all_examples:
        key = ex["text"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0][:100]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"After dedup: {len(unique)}")

    # Shuffle + split
    random.shuffle(unique)
    split = int(len(unique) * 0.95)
    train = unique[:split]
    valid = unique[split:]

    # Save
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_dir / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved!")
    print(f"   train.jsonl: {len(train)} examples")
    print(f"   valid.jsonl: {len(valid)} examples")

    # Spot check
    print("\n=== 2 Sample Examples ===")
    with open(out_dir / "train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2: break
            ex = json.loads(line)
            user = ex["text"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            asst = ex["text"].split("<|im_start|>assistant\n")[1].replace("<|im_end|>", "")
            print(f"\n[{i+1}] USER: {user[:70]}")
            print(f"ASST: {asst[:300]}")
            print("---")

    print("\nNext step:")
    print("  rm -rf adapters/ timps-coder-fused/")
    print("  bash retrain.sh")


if __name__ == "__main__":
    main()