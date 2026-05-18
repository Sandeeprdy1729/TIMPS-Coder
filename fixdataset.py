"""
TIMPS-Coder — Fix Dataset: Extract Explanations from Trajectory
================================================================
The REAL problem: your `output` field has code-only answers,
but `trajectory` contains rich root-cause explanations.

This script rebuilds train.jsonl with:
  output = [root_cause explanation] + [fixed code]

This is the correct fix. Run this, then retrain.

Usage: python fix_dataset.py
"""

import json
import re
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

SFT_FILE  = "data/raw/sft_train.jsonl"
RLEF_FILE = "data/raw/rlef_train.jsonl"
OUT_DIR   = "data/processed"
VALID_SPLIT = 0.05

SYSTEM = (
    "You are TIMPS-Coder, an expert coding assistant. "
    "When asked to fix a bug, first explain the root cause clearly in plain English, "
    "then provide the corrected code."
)


# ── Trajectory Extraction ────────────────────────────────────

def extract_root_cause(trajectory: str) -> str:
    """Pull root cause explanation from trajectory field."""
    text = trajectory

    # Try "Root cause:" section
    match = re.search(r'Root cause:\s*\n?(.*?)(?:\nFixed code|```|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        rc = match.group(1).strip()
        # Remove code blocks inside root cause
        rc = re.sub(r'```[^`]*```', '', rc, flags=re.DOTALL).strip()
        if len(rc) > 60:
            return rc[:800]  # Cap length

    # Try "Problem:" section as fallback
    match = re.search(r'Problem:\s*\n?(.*?)(?:\nBuggy code|Root cause|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        prob = match.group(1).strip()
        prob = re.sub(r'```[^`]*```', '', prob, flags=re.DOTALL).strip()
        if len(prob) > 60:
            return prob[:600]

    return ""


def extract_fixed_code(trajectory: str, fallback_output: str) -> str:
    """Extract the fixed code block from trajectory."""
    # Try "Fixed code" section
    match = re.search(r'Fixed code[^:]*:\s*\n?```[^\n]*\n(.*?)```', trajectory, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to output field
    return fallback_output.strip()


def build_rich_output(trajectory: str, raw_output: str, lang: str = "") -> str:
    """
    Combine root cause explanation + fixed code into one rich answer.
    Format:
      The issue is ... [prose explanation] ...

      Fixed code:
      ```java
      ...
      ```
    """
    root_cause = extract_root_cause(trajectory)
    fixed_code = extract_fixed_code(trajectory, raw_output)

    # Determine language for code fence
    lang_hint = lang.lower() if lang else ""
    if not lang_hint:
        # Guess from code content
        if "def " in fixed_code or "import " in fixed_code and "java" not in fixed_code:
            lang_hint = "python"
        elif "public class" in fixed_code or "@" in fixed_code:
            lang_hint = "java"
        elif "function" in fixed_code or "const " in fixed_code or "=>" in fixed_code:
            lang_hint = "javascript"

    parts = []

    if root_cause:
        parts.append(root_cause)

    if fixed_code:
        parts.append(f"**Fixed code:**\n```{lang_hint}\n{fixed_code}\n```")

    if not parts:
        return raw_output  # Last resort

    return "\n\n".join(parts)


# ── Filtering ────────────────────────────────────────────────

def is_good_example(instruction: str, output: str) -> bool:
    """Filter out low-quality examples."""
    if len(output) < 100:
        return False
    if len(instruction.strip()) < 20:
        return False
    # Must have either prose OR a code block (not just a one-liner)
    has_code_block = "```" in output
    has_prose = any(
        phrase in output.lower()
        for phrase in [
            "because", "the issue", "the problem", "this happens",
            "null because", "spring only", "you need to", "instead",
            "the reason", "fixed code", "this means", "note that",
            "when you", "the error", "root cause"
        ]
    )
    return has_code_block or has_prose


# ── Format ───────────────────────────────────────────────────

def format_chatml(instruction: str, output: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


# ── Loaders ──────────────────────────────────────────────────

def load_sft(path: str) -> list:
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            instruction = d.get("instruction", "").strip()
            raw_output  = d.get("output", "").strip()
            trajectory  = d.get("trajectory", "")
            lang        = d.get("metadata", {}).get("language", "")

            # Build rich output using trajectory explanation
            rich_output = build_rich_output(trajectory, raw_output, lang)

            if not is_good_example(instruction, rich_output):
                skipped += 1
                continue

            examples.append({"text": format_chatml(instruction, rich_output)})

    print(f"  SFT: {len(examples)} kept, {skipped} skipped")
    return examples


def load_rlef(path: str, max_examples: int = 4000) -> list:
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get("result", "").lower() != "pass":
                continue

            instruction = d.get("instruction", "").strip()
            response    = d.get("response", "").strip()
            reasoning   = d.get("reasoning", "").strip()

            if reasoning and len(reasoning) > 60:
                output = f"{reasoning}\n\n**Fixed code:**\n```\n{response}\n```"
            else:
                output = response

            if not is_good_example(instruction, output):
                skipped += 1
                continue

            examples.append({"text": format_chatml(instruction, output)})

    random.shuffle(examples)
    examples = examples[:max_examples]
    print(f"  RLEF: {len(examples)} kept, {skipped} skipped (pass-only, capped at {max_examples})")
    return examples


# ── Main ─────────────────────────────────────────────────────

def main():
    print("\n🔧 TIMPS-Coder Dataset Rebuilder")
    print("=" * 45)
    print("Strategy: Extract root-cause from trajectory → richer outputs\n")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading + rebuilding SFT data...")
    sft = load_sft(SFT_FILE)

    print("[2/3] Loading RLEF (pass-only)...")
    rlef = load_rlef(RLEF_FILE)

    all_examples = sft + rlef
    random.shuffle(all_examples)
    print(f"\n[3/3] Total: {len(all_examples)} examples")

    split = int(len(all_examples) * (1 - VALID_SPLIT))
    train = all_examples[:split]
    valid = all_examples[split:]

    with open(f"{OUT_DIR}/train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(f"{OUT_DIR}/valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train)} | Valid: {len(valid)}")

    # Show 2 sample outputs so you can verify quality
    print("\n✅ Sample outputs (verify these look good before retraining):")
    print("=" * 55)
    with open(f"{OUT_DIR}/train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            ex = json.loads(line)
            text = ex["text"]
            # Show just the assistant part
            assistant_part = text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>","")
            user_part = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            print(f"\n[Example {i+1}]")
            print(f"USER: {user_part[:80]}")
            print(f"ASSISTANT:\n{assistant_part[:500]}")
            print("-" * 55)

    print(f"\n📁 Saved to {OUT_DIR}/")
    print("\n⚠️  Now DELETE your old adapters and retrain:")
    print("   rm -rf adapters/ timps-coder-fused/")
    print("   bash 2_train_sft.sh")


if __name__ == "__main__":
    main()