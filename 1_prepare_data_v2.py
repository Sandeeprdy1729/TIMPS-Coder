#!/usr/bin/env python3
"""
TIMPS-Coder v2 — Advanced Dataset Preparation
==============================================
Builds a high-signal, agentic training dataset from:

  1. SWE-bench/SWE-bench_Verified        — Real GitHub issue + patch pairs (~400)
  2. TIGER-Lab/SWE-Next-SFT-Trajectories — Expert agentic edit traces (~2K)
  3. newfacade/LeetCodeDataset           — Algorithm problem solving (~2.5K)
  4. WaltonFuture/agentic-sft-new        — Tool use + bash agent traces (~3K)
  5. openai/openai_humaneval             — Code completion (fallback, ~164)
  6. Existing TIMPS v1 processed data    — Identity + bug-fix baseline

What makes v2 different from generic code models:
  - Real codebase navigation (SWE-bench style)
  - Multi-step agentic reasoning traces
  - Explicit THINK → ACT → VERIFY pattern
  - Tool use and bash command execution
  - Algorithmic depth + complexity analysis

Output:  data/processed/train.jsonl + data/processed/valid.jsonl
Usage:   pip install datasets && python3 1_prepare_data_v2.py
"""

import json
import random
import os
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    print("❌  Missing dependency:  pip install datasets")
    raise SystemExit(1)

random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
OUT_DIR      = "data/processed"
VALID_SPLIT  = 0.05        # 5% held-out validation
MAX_TEXT_LEN = 6000        # char cap ≈ 2048 tokens (safe for seq-len 2048)

# v2 system prompt — the identity that makes TIMPS-Coder different
SYSTEM_V2 = """You are TIMPS-Coder v2, an agentic software engineer built by Sandeep Reddy (TIMPS).

Unlike generic assistants you are trained on real GitHub issue resolutions, expert \
agent execution traces, and competitive algorithm problems. You go beyond simple \
bug fixes — you reason through root causes, navigate codebases, and write \
production-ready code.

For every task follow this pattern:
  THINK  — Reason through the problem step-by-step (root cause / approach)
  ACT    — Write complete, correct, well-formatted code
  VERIFY — Address edge cases, complexity, and potential follow-up issues"""
# ─────────────────────────────────────────────────────────────────────────────

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ── Formatting helpers ────────────────────────────────────────────────────────

def chatml(instruction: str, response: str, system: str = SYSTEM_V2) -> str:
    """Wrap a single instruction/response pair in ChatML format."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{response.strip()}<|im_end|>"
    )


def is_valid(text: str) -> bool:
    """Quality gate: reject empty, too-short, too-long, or prose-only examples."""
    if len(text) < 120 or len(text) > MAX_TEXT_LEN:
        return False
    code_signals = [
        "```", "def ", "class ", "function ", "const ", "let ", "var ",
        "import ", "return ", "if (", "for (", "while (", "public ", "private ",
        "fn ", "func ", "package ", "#include", "void ", "int main", "async ",
        "await ", "SELECT ", "INSERT ", "CREATE ", "$ ", "git ", "pip ", "npm ",
    ]
    return any(sig in text for sig in code_signals)


def safe_load(ds_name: str, split: str = "train", streaming: bool = False, **kwargs):
    """Load a HuggingFace dataset, return None on failure."""
    try:
        ds = load_dataset(ds_name, split=split, streaming=streaming, **kwargs)
        return ds
    except Exception as e:
        print(f"    ⚠️  Could not load {ds_name}: {e}")
        return None


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_swebench(max_samples: int = 400) -> list[dict]:
    """
    SWE-bench Verified — Real GitHub issues paired with validated patches.
    Teaches the model to: read an issue, understand a codebase, produce a diff.
    """
    print("\n[1/5] SWE-bench/SWE-bench_Verified  (real GitHub issue → patch)")
    examples = []

    # SWE-bench is small (~500 rows), no streaming needed
    ds = safe_load("SWE-bench/SWE-bench_Verified", split="test")
    if ds is None:
        ds = safe_load("princeton-nlp/SWE-bench_Verified", split="test")
    if ds is None:
        print("    Skipped.")
        return examples

    for row in ds:
        repo    = row.get("repo", "unknown/repo")
        problem = (row.get("problem_statement") or "").strip()
        patch   = (row.get("patch") or "").strip()
        hints   = (row.get("hints_text") or "").strip()

        if not problem or not patch:
            continue

        hint_section = f"\n\n**Hints from maintainers:**\n{hints[:300]}" if hints else ""

        instruction = (
            f"**Repository:** `{repo}`\n\n"
            f"**GitHub Issue:**\n{problem[:900]}"
            f"{hint_section}"
        )

        response = (
            f"**THINK:**\n"
            f"Analyzing the issue in `{repo}`. The bug report indicates a problem "
            f"in the code path described above. Let me trace through the logic to "
            f"identify the root cause.\n\n"
            f"**ACT — Patch:**\n"
            f"```diff\n{patch[:1400]}\n```\n\n"
            f"**VERIFY:**\n"
            f"This patch is minimal and targeted. It directly addresses the reported "
            f"behaviour without introducing regressions. The existing test suite should "
            f"be run with `python -m pytest` to confirm the fix."
        )

        text = chatml(instruction, response)
        if is_valid(text):
            examples.append({"text": text})

        if len(examples) >= max_samples:
            break

    print(f"    ✓  {len(examples):,} examples")
    return examples


def load_swe_next(max_samples: int = 2000) -> list[dict]:
    """
    TIGER-Lab/SWE-Next-SFT-Trajectories — ShareGPT multi-turn agentic traces.
    Teaches the model to: navigate repos, issue bash commands, iteratively fix code.
    """
    print("\n[2/5] TIGER-Lab/SWE-Next-SFT-Trajectories  (agentic multi-turn traces)")
    examples = []

    # ~200MB already cached — load normally (not streaming)
    ds = safe_load("TIGER-Lab/SWE-Next-SFT-Trajectories", split="train")
    if ds is None:
        print("    Skipped.")
        return examples

    for row in ds:
        convs = row.get("conversations") or row.get("messages") or []
        if not convs:
            continue

        parts = [f"<|im_start|>system\n{SYSTEM_V2}<|im_end|>"]
        valid = True

        for turn in convs:
            if not isinstance(turn, dict):
                valid = False
                break
            role    = (turn.get("from") or turn.get("role") or "").lower()
            content = (turn.get("value") or turn.get("content") or "").strip()
            if not content:
                continue
            if role in ("human", "user", "tool"):
                # 'tool' = execution output — treat as user-side context
                parts.append(f"<|im_start|>user\n{content[:700]}<|im_end|>")
            elif role in ("gpt", "assistant"):
                parts.append(f"<|im_start|>assistant\n{content[:900]}<|im_end|>")
            elif role == "system":
                pass  # We inject our own system prompt
            # any other unknown role — silently skip, don't reject whole trace

        if not valid or len(parts) < 3:
            continue

        text = "\n".join(parts)

        # Truncate over-long multi-turn trajectories — keep first 2 + last 2 turns
        if len(text) > MAX_TEXT_LEN and len(parts) > 5:
            text = "\n".join(parts[:3] + parts[-2:])

        if is_valid(text) and len(text) <= MAX_TEXT_LEN:
            examples.append({"text": text})

        if len(examples) >= max_samples:
            break

    print(f"    ✓  {len(examples):,} examples")
    return examples


def load_leetcode(max_samples: int = 2500) -> list[dict]:
    """
    LeetCode dataset — Algorithm problems with solutions.
    Teaches the model to: structure reasoning, write efficient algorithms, analyse complexity.
    """
    print("\n[3/5] LeetCode / HumanEval  (algorithmic problem solving)")
    examples = []

    # ~96MB already cached — load normally
    ds = safe_load("newfacade/LeetCodeDataset", split="train")
    if ds is not None:
        for row in ds:
            title = (row.get("task_id") or row.get("title") or row.get("problem_title") or "Problem").strip()
            desc  = (
                row.get("query") or row.get("problem_description") or
                row.get("description") or row.get("content") or
                row.get("question") or ""
            ).strip()
            sol   = (
                row.get("response") or row.get("solution") or row.get("python_solution") or
                row.get("code") or row.get("answer") or ""
            ).strip()
            diff  = row.get("difficulty", "Medium")
            tags  = row.get("tags", row.get("topic_tags", ""))
            if isinstance(tags, list):
                tags = ", ".join(tags[:4])

            if not desc or not sol:
                continue

            instruction = (
                f"**LeetCode — {title}** ({diff})\n\n"
                f"Tags: {tags}\n\n"
                f"{desc[:700]}"
            )
            response = (
                f"**THINK:**\n"
                f"This is a {diff} problem. Let me identify the key insight:\n"
                f"- Understand the constraints and expected behaviour\n"
                f"- Choose the optimal data structure / algorithm pattern\n"
                f"- Plan the implementation before writing code\n\n"
                f"**ACT:**\n"
                f"```python\n{sol[:900]}\n```\n\n"
                f"**VERIFY:**\n"
                f"Check edge cases: empty input, single element, duplicates, "
                f"overflow conditions. Analyse time and space complexity."
            )
            text = chatml(instruction, response)
            if is_valid(text):
                examples.append({"text": text})
            if len(examples) >= max_samples:
                break

    # Fallback: openai/openai_humaneval (always public, tiny ~96KB)
    if len(examples) < 100:
        print("    → Falling back to openai/openai_humaneval")
        ds2 = safe_load("openai/openai_humaneval", split="test")
        if ds2 is not None:
            for row in ds2:
                prompt    = (row.get("prompt") or "").strip()
                canonical = (row.get("canonical_solution") or "").strip()
                entry     = row.get("entry_point", "solve")
                if not prompt or not canonical:
                    continue
                instruction = (
                    f"Complete this Python function and explain your approach:\n\n"
                    f"```python\n{prompt}\n```"
                )
                response = (
                    f"**THINK:**\n"
                    f"The function `{entry}` needs to satisfy the docstring contract. "
                    f"Let me reason through the logic step by step.\n\n"
                    f"**ACT:**\n"
                    f"```python\n{prompt}{canonical}\n```\n\n"
                    f"**VERIFY:**\n"
                    f"The solution handles the examples in the docstring. "
                    f"Edge cases considered: empty inputs, boundary values."
                )
                text = chatml(instruction, response)
                if is_valid(text):
                    examples.append({"text": text})

    print(f"    ✓  {len(examples):,} examples")
    return examples


def load_agentic(max_samples: int = 3000) -> list[dict]:
    """
    Agentic coding traces — tool use, bash, multi-step task execution.
    Teaches the model to: call tools, run commands, plan multi-step solutions.
    """
    print("\n[4/5] Agentic coding trajectories  (tool use + bash + multi-step)")
    examples = []

    # Use streaming=True for all agentic sources — avoids downloading multi-GB
    # archives that fill the disk. We only pull the rows we need.
    sources = [
        ("WaltonFuture/agentic-sft-new",          "train",  True),
        ("DeepNLP/Coding-Agent-Github-2025-Feb",   "train",  False),  # small, no streaming needed
        ("AlienKevin/SWE-ZERO-12M-trajectories",   "train",  True),
    ]

    for ds_name, split, use_streaming in sources:
        if len(examples) >= max_samples:
            break

        ds = safe_load(ds_name, split=split, streaming=use_streaming)
        if ds is None:
            continue

        loaded_from_source = 0
        quota = max_samples - len(examples)

        for row in ds:
            # Handle ShareGPT conversation format
            convs = (
                row.get("conversations") or row.get("messages") or
                row.get("trajectory") or []
            )

            if convs and isinstance(convs, list):
                parts = [f"<|im_start|>system\n{SYSTEM_V2}<|im_end|>"]
                valid = True
                for turn in convs:
                    if not isinstance(turn, dict):
                        valid = False
                        break
                    role    = (turn.get("from") or turn.get("role") or "").lower()
                    content = (turn.get("value") or turn.get("content") or "").strip()
                    if not content:
                        continue
                    if role in ("human", "user", "tool"):
                        parts.append(f"<|im_start|>user\n{content[:600]}<|im_end|>")
                    elif role in ("gpt", "assistant"):
                        parts.append(f"<|im_start|>assistant\n{content[:800]}<|im_end|>")
                    elif role == "system":
                        pass  # skip, we inject our own

                if valid and len(parts) >= 3:
                    text = "\n".join(parts)
                    if len(text) > MAX_TEXT_LEN and len(parts) > 5:
                        text = "\n".join(parts[:3] + parts[-2:])
                    if is_valid(text) and len(text) <= MAX_TEXT_LEN:
                        examples.append({"text": text})
                        loaded_from_source += 1

            # Handle simple instruction/output format
            elif row.get("instruction") and row.get("output"):
                text = chatml(str(row["instruction"])[:500], str(row["output"])[:800])
                if is_valid(text):
                    examples.append({"text": text})
                    loaded_from_source += 1

            if loaded_from_source >= quota:
                break

        if loaded_from_source > 0:
            print(f"    ✓  +{loaded_from_source:,} from {ds_name}")

    print(f"    ✓  {len(examples):,} agentic examples total")
    return examples


def load_existing() -> list[dict]:
    """
    Preserve v1 TIMPS training data — maintains identity & existing bug-fix strength.
    """
    print("\n[5/5] Existing TIMPS v1 data  (identity + bug-fix baseline)")
    examples = []

    raw_paths = [
        "data/raw/sft_train.jsonl",
        "data/raw/rlef_train.jsonl",
    ]
    processed_paths = [
        "data/processed/train.jsonl",
        "data/processed/valid.jsonl",
    ]

    for path in raw_paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("text"):
                    examples.append({"text": d["text"]})
                elif d.get("instruction") and d.get("output"):
                    examples.append({"text": chatml(d["instruction"], d["output"])})

    for path in processed_paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("text"):
                    examples.append({"text": d["text"]})

    print(f"    ✓  {len(examples):,} examples")
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  TIMPS-Coder v2 — Dataset Builder")
    print("=" * 65)

    swebench  = load_swebench(max_samples=400)
    # SWE-Next removed: assistant turns are short tool-calls that cause catastrophic forgetting
    leetcode  = load_leetcode(max_samples=2500)
    agentic   = load_agentic(max_samples=3000)
    existing  = load_existing()

    all_examples = swebench + leetcode + agentic + existing

    # Deduplicate by user-message content (skip the common system prompt header)
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in all_examples:
        text = ex["text"]
        user_pos = text.find("<|im_start|>user\n")
        key = text[user_pos:user_pos + 150] if user_pos != -1 else text[:150]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    random.shuffle(unique)

    n_valid   = max(100, int(len(unique) * VALID_SPLIT))
    valid_set = unique[:n_valid]
    train_set = unique[n_valid:]

    # Disk space guard before writing
    import shutil
    free_bytes = shutil.disk_usage(OUT_DIR).free
    free_gb    = free_bytes / (1024 ** 3)
    if free_gb < 0.5:
        print(f"\n❌  Only {free_gb:.1f} GB free on disk — not enough to write output.")
        print("    Free up space with:")
        print("      rm -rf ~/.cache/huggingface/hub/datasets--*")
        raise SystemExit(1)

    # Write output
    train_path = f"{OUT_DIR}/train.jsonl"
    valid_path = f"{OUT_DIR}/valid.jsonl"

    with open(train_path, "w") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 65}")
    print(f"  Dataset Summary — TIMPS-Coder v2")
    print(f"{'=' * 65}")
    print(f"  SWE-bench (real GitHub bug-fix)  : {len(swebench):>6,}")

    print(f"  LeetCode  (algorithm problems)   : {len(leetcode):>6,}")
    print(f"  Agentic   (tool use + bash)      : {len(agentic):>6,}")
    print(f"  TIMPS v1  (identity baseline)    : {len(existing):>6,}")
    print(f"  {'─' * 40}")
    print(f"  Total unique examples            : {len(unique):>6,}")
    print(f"  Train                            : {len(train_set):>6,}")
    print(f"  Validation                       : {len(valid_set):>6,}")
    print(f"\n  ✅  Saved → {train_path}")
    print(f"  ✅  Saved → {valid_path}")
    print(f"\n  Next step:  bash 2_train_sft.sh")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
