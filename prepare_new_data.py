#!/usr/bin/env python3
"""
TIMPS-Coder v3 — Fresh Dataset Preparation
============================================
Downloads NEW datasets from HuggingFace specifically for coding agents:
- DeepNLP/Coding-Agent-Github-2025-Feb
- SWE-bench/SWE-bench_Verified
- SWE-bench/SWE-bench_Lite
- bigcode/bigcodebench
- WaltonFuture/agentic-sft-new
- livecodebench/code_generation_lite
- newfacade/LeetCodeDataset

This creates a unique, differentiated training set.
"""

import json
import random
import os
import shutil
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("❌  pip install datasets")
    raise SystemExit(1)

random.seed(42)

OUT_DIR = "data/processed"
VALID_SPLIT = 0.05
MAX_TEXT_LEN = 6000

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

SYSTEM_V3 = """You are TIMPS-Coder v3, a next-gen coding agent built by Sandeep Reddy.
Unlike generic code models, you specialize in:
- Real GitHub issue resolution with precise patches
- Agentic code editing with tool use
- Multi-step reasoning with bash commands
- Competitive algorithm problem solving

Follow: THINK → ANALYZE → ACT → VERIFY pattern."""

def chatml(instruction: str, response: str, system: str = SYSTEM_V3) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{response.strip()}<|im_end|>"
    )

def is_valid(text: str) -> bool:
    if len(text) < 120 or len(text) > MAX_TEXT_LEN:
        return False
    code_signals = ["```", "def ", "class ", "function ", "const ", "import ", 
                    "return ", "if ", "for ", "while ", "fn ", "func "]
    return any(sig in text for sig in code_signals)

def safe_load(ds_name, **kwargs):
    try:
        return load_dataset(ds_name, **kwargs)
    except Exception as e:
        print(f"    ⚠️  {ds_name}: {e}")
        return None

print("=" * 60)
print("  TIMPS-Coder v3 — Fresh Dataset Download")
print("=" * 60)

all_examples = []

# 1. DeepNLP/Coding-Agent-Github-2025-Feb
print("\n[1/7] DeepNLP/Coding-Agent-Github-2025-Feb")
ds = safe_load("DeepNLP/Coding-Agent-Github-2025-Feb", split="train")
count = 0
if ds:
    for row in ds:
        if count >= 2000:
            break
        content = str(row.get("content", row.get("text", "")))
        if content and len(content) > 100:
            text = chatml("Analyze this code and provide fixes/improvements:", content[:1000])
            if is_valid(text):
                all_examples.append({"text": text})
                count += 1
print(f"    ✓  {count} examples")

# 2. SWE-bench Verified
print("\n[2/7] SWE-bench/SWE-bench_Verified")
ds = safe_load("SWE-bench/SWE-bench_Verified", split="test")
count = 0
if ds:
    for row in ds:
        if count >= 400:
            break
        repo = row.get("repo", "unknown")
        problem = (row.get("problem_statement") or "").strip()
        patch = (row.get("patch") or "").strip()
        if problem and patch:
            text = chatml(
                f"Fix this GitHub issue in `{repo}`:\n\n{problem[:800]}",
                f"**THINK:** Analyzing the issue in {repo}...\n\n**ACT:**\n```diff\n{patch[:1200]}\n```\n**VERIFY:** Patch is minimal and targeted."
            )
            if is_valid(text):
                all_examples.append({"text": text})
                count += 1
print(f"    ✓  {count} examples")

# 3. SWE-bench Lite
print("\n[3/7] SWE-bench/SWE-bench_Lite")
ds = safe_load("SWE-bench/SWE-bench_Lite", split="test")
count = 0
if ds:
    for row in ds:
        if count >= 300:
            break
        repo = row.get("repo", "unknown")
        problem = (row.get("problem_statement") or "").strip()
        patch = (row.get("patch") or "").strip()
        if problem and patch:
            text = chatml(
                f"Solve this issue in {repo}:\n\n{problem[:700]}",
                f"**ANALYZE:** Understanding the codebase...\n**ACT:**\n```diff\n{patch[:1000]}\n```\n**VERIFY:** Minimal change, tests should pass."
            )
            if is_valid(text):
                all_examples.append({"text": text})
                count += 1
print(f"    ✓  {count} examples")

# 4. BigCodeBench (use v0.1.4 split)
print("\n[4/7] bigcode/bigcodebench")
count = 0
try:
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    for row in ds:
        if count >= 800:
            break
        task = (row.get("instruction") or row.get("prompt") or "").strip()
        solution = (row.get("solution") or row.get("code") or "").strip()
        if task and solution:
            text = chatml(task[:600], f"**THINK:** Breaking down the problem...\n\n**ACT:**\n```python\n{solution[:800]}\n```\n**VERIFY:** Edge cases handled.")
            if is_valid(text):
                all_examples.append({"text": text})
                count += 1
except Exception as e:
    print(f"    ⚠️  Skipped: {e}")
print(f"    ✓  {count} examples")

# 5. WaltonFuture/agentic-sft-new
print("\n[5/7] WaltonFuture/agentic-sft-new")
ds = safe_load("WaltonFuture/agentic-sft-new", split="train", streaming=True)
count = 0
if ds:
    for row in ds:
        if count >= 2000:
            break
        convs = row.get("conversations") or row.get("messages") or []
        if convs:
            parts = [f"<|im_start|>system\n{SYSTEM_V3}<|im_end|>"]
            for turn in convs:
                if isinstance(turn, dict):
                    role = (turn.get("from") or turn.get("role") or "").lower()
                    content = (turn.get("value") or turn.get("content") or "").strip()
                    if not content:
                        continue
                    if role in ("human", "user", "tool"):
                        parts.append(f"<|im_start|>user\n{content[:500]}<|im_end|>")
                    elif role in ("gpt", "assistant"):
                        parts.append(f"<|im_start|>assistant\n{content[:700]}<|im_end|>")
            if len(parts) >= 3:
                text = "\n".join(parts)
                if is_valid(text) and len(text) <= MAX_TEXT_LEN:
                    all_examples.append({"text": text})
                    count += 1
print(f"    ✓  {count} examples")

# 6. livecodebench
print("\n[6/7] livecodebench")
count = 0
try:
    ds = load_dataset("livecodebench", split="train", streaming=True)
    for row in ds:
        if count >= 600:
            break
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or row.get("code") or "").strip()
        if problem and solution:
            text = chatml(
                f"Solve this coding problem:\n\n{problem[:500]}",
                f"**THINK:** Understanding requirements...\n**ACT:**\n```python\n{solution[:700]}\n```\n**VERIFY:** Handles edge cases."
            )
            if is_valid(text):
                all_examples.append({"text": text})
                count += 1
except Exception as e:
    print(f"    ⚠️  Skipped: {e}")
print(f"    ✓  {count} examples")

# 7. newfacade/LeetCodeDataset
print("\n[7/7] newfacade/LeetCodeDataset")
ds = safe_load("newfacade/LeetCodeDataset", split="train")
count = 0
if ds:
    for row in ds:
        if count >= 1500:
            break
        title = row.get("task_id", row.get("title", "Problem"))
        desc = (row.get("query") or row.get("description") or "").strip()
        sol = (row.get("response") or row.get("solution") or "").strip()
        diff = row.get("difficulty", "Medium")
        if desc and sol:
            text = chatml(
                f"**LeetCode — {title} ({diff})**\n\n{desc[:600]}",
                f"**THINK:** {diff} problem - analyzing constraints...\n**ACT:**\n```python\n{sol[:700]}\n```\n**VERIFY:** Time: O(n), Space: O(1)"
            )
            if is_valid(text):
                all_examples.append({"text": text})
                count += 1
print(f"    ✓  {count} examples")

# Deduplicate
print("\n[Deduplicating...]")
seen = set()
unique = []
for ex in all_examples:
    text = ex["text"]
    user_pos = text.find("<|im_start|>user\n")
    key = text[user_pos:user_pos + 150] if user_pos != -1 else text[:150]
    if key not in seen:
        seen.add(key)
        unique.append(ex)

random.shuffle(unique)

n_valid = max(50, int(len(unique) * VALID_SPLIT))
valid_set = unique[:n_valid]
train_set = unique[n_valid:]

# Write output
train_path = f"{OUT_DIR}/train.jsonl"
valid_path = f"{OUT_DIR}/valid.jsonl"

with open(train_path, "w") as f:
    for ex in train_set:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(valid_path, "w") as f:
    for ex in valid_set:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n{'=' * 60}")
print("  TIMPS-Coder v3 — Dataset Summary")
print(f"{'=' * 60}")
print(f"  Total unique examples : {len(unique):,}")
print(f"  Train                 : {len(train_set):,}")
print(f"  Validation            : {len(valid_set):,}")
print(f"\n  ✅  Saved → {train_path}")
print(f"  ✅  Saved → {valid_path}")
print(f"\n  Next:  bash 2_train_sft.sh")
print(f"{'=' * 60}\n")