#!/usr/bin/env python3
"""
TIMPS-Coder — Diagnose + Fix Dataset
======================================
Run this on your Mac FIRST to see what's wrong with your train.jsonl,
then it auto-fixes it.

Usage: python3 fix_fences.py
"""

import json, re, random
from pathlib import Path

random.seed(42)

TRAIN_IN  = "data/processed/train.jsonl"
VALID_IN  = "data/processed/valid.jsonl"
TRAIN_OUT = "data/processed/train_fixed.jsonl"
VALID_OUT = "data/processed/valid_fixed.jsonl"

# ── Diagnose ─────────────────────────────────────────────────
def diagnose(path):
    total = has_fence = no_fence = short_asst = 0
    
    print(f"\nDiagnosing {path}...")
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            text = ex["text"]
            asst = text.split("<|im_start|>assistant\n")[1].replace("<|im_end|>","").strip()
            total += 1
            if "```" in asst:
                has_fence += 1
            else:
                no_fence += 1
            if len(asst) < 100:
                short_asst += 1
    
    print(f"  Total:           {total}")
    print(f"  Has ``` fences:  {has_fence} ({100*has_fence/total:.1f}%)")
    print(f"  NO ``` fences:   {no_fence} ({100*no_fence/total:.1f}%)  ← this is your problem")
    print(f"  Short (<100ch):  {short_asst} ({100*short_asst/total:.1f}%)")
    return total, has_fence, no_fence

# ── Detect language from code ─────────────────────────────────
def detect_lang(code: str) -> str:
    if "def " in code or "import " in code and "java" not in code.lower():
        return "python"
    if "public " in code or "@Override" in code or "System.out" in code:
        return "java"
    if "function " in code or "const " in code or "=>" in code or "console.log" in code:
        return "javascript"
    if "#include" in code or "std::" in code or "cout" in code:
        return "cpp"
    if "fn " in code and "let " in code:
        return "rust"
    if "func " in code and ":=" in code:
        return "go"
    return ""

# ── Fix a single assistant response ──────────────────────────
def fix_assistant(asst: str) -> str:
    """
    If response has no ``` fence:
    - Split into prose (explanation) and code parts
    - Wrap code in appropriate fence
    """
    if "```" in asst:
        return asst  # Already has fence, nothing to do
    
    lines = asst.split("\n")
    prose_lines = []
    code_lines  = []
    
    CODE_INDICATORS = [
        "def ", "class ", "public ", "private ", "return ",
        "import ", "from ", "function ", "const ", "let ", "var ",
        "if (", "for (", "while (", "    ", "\t",
        "System.out", "print(", "console.log", "#include",
        "=>", "->", "::", "@Override", "@Component",
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty line — goes to whichever group we're currently in
            if code_lines:
                code_lines.append(line)
            else:
                prose_lines.append(line)
            continue
        
        is_code = (
            any(ind in line for ind in CODE_INDICATORS) or
            (line.startswith("    ") and len(stripped) > 3) or
            re.match(r'^[a-z_]+\(', stripped) or   # function call
            re.match(r'^\s*[{}()\[\];]', stripped)  # brackets
        )
        
        # Short single-word lines mixed with prose — treat as prose
        if is_code and len(stripped) > 5:
            code_lines.append(line)
        else:
            if code_lines:
                code_lines.append(line)  # continuation
            else:
                prose_lines.append(line)
    
    # Clean up
    while prose_lines and not prose_lines[-1].strip(): prose_lines.pop()
    while code_lines  and not code_lines[0].strip():  code_lines.pop(0)
    while code_lines  and not code_lines[-1].strip():  code_lines.pop()
    
    prose = "\n".join(prose_lines).strip()
    code  = "\n".join(code_lines).strip()
    
    if not code:
        return asst  # Can't fix — no code detected at all
    
    lang = detect_lang(code)
    parts = []
    if prose:
        parts.append(prose)
    parts.append(f"```{lang}\n{code}\n```")
    return "\n\n".join(parts)

# ── Filter + fix ─────────────────────────────────────────────
def fix_file(in_path: str, out_path: str) -> tuple:
    kept = fixed = dropped = 0
    
    with open(in_path)  as fin, \
         open(out_path, "w") as fout:
        
        for line in fin:
            ex = json.loads(line)
            text = ex["text"]
            
            # Extract parts
            try:
                system = text.split("<|im_start|>system\n")[1].split("<|im_end|>")[0]
                user   = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
                asst   = text.split("<|im_start|>assistant\n")[1].replace("<|im_end|>","").strip()
            except IndexError:
                dropped += 1
                continue
            
            # Fix fences
            fixed_asst = fix_assistant(asst)
            
            # Quality gate AFTER fixing
            if "```" not in fixed_asst:
                dropped += 1
                continue
            if len(fixed_asst.strip()) < 80:
                dropped += 1
                continue
            
            # Check prose exists (text outside code blocks)
            prose_only = re.sub(r"```.*?```", "", fixed_asst, flags=re.DOTALL).strip()
            if len(prose_only) < 30:
                dropped += 1
                continue
            
            if fixed_asst != asst:
                fixed += 1
            else:
                kept += 1
            
            # Rebuild clean text
            new_text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{fixed_asst}<|im_end|>"
            )
            fout.write(json.dumps({"text": new_text}, ensure_ascii=False) + "\n")
    
    return kept, fixed, dropped

# ── Main ─────────────────────────────────────────────────────
def main():
    print("🔍 TIMPS-Coder Dataset Fence Fixer")
    print("=" * 45)
    
    # Diagnose first
    t_total, t_fence, t_nofence = diagnose(TRAIN_IN)
    v_total, v_fence, v_nofence = diagnose(VALID_IN)
    
    if t_nofence == 0:
        print("\n✅ Dataset already has 100% code fences — problem is elsewhere.")
        print("   The issue might be the model itself. Try: --temp 0.05 --max-tokens 600")
        return
    
    print(f"\n🔧 Fixing {t_nofence + v_nofence} examples without code fences...")
    
    # Fix train
    tk, tf, td = fix_file(TRAIN_IN, TRAIN_OUT)
    print(f"\nTrain: {tk} already-ok + {tf} fixed = {tk+tf} kept, {td} dropped")
    
    # Fix valid
    vk, vf, vd = fix_file(VALID_IN, VALID_OUT)
    print(f"Valid: {vk} already-ok + {vf} fixed = {vk+vf} kept, {vd} dropped")
    
    # Replace originals
    import shutil
    shutil.move(TRAIN_OUT, TRAIN_IN)
    shutil.move(VALID_OUT, VALID_IN)
    print(f"\n✅ Files replaced in-place.")
    
    # Final verify
    print("\nVerifying fix...")
    _, final_fence, final_nofence = diagnose(TRAIN_IN)
    if final_nofence == 0:
        print("\n🎉 PERFECT — 100% of examples now have ``` code fences")
        print("\nNow retrain:")
        print("  rm -rf adapters/ timps-coder-fused/")
        print("  bash retrain.sh")
    else:
        print(f"\n⚠️  {final_nofence} examples still have no fence — they were dropped during quality gate")
        pct = 100 * final_fence / (final_fence + final_nofence)
        print(f"   {pct:.1f}% have fences — this is good enough to proceed.")
        print("\n  rm -rf adapters/ timps-coder-fused/")
        print("  bash retrain.sh")

    # Show 2 examples to verify
    print("\n=== 2 SAMPLE EXAMPLES after fix ===")
    with open(TRAIN_IN) as f:
        for i, line in enumerate(f):
            if i >= 2: break
            ex = json.loads(line)
            asst = ex["text"].split("<|im_start|>assistant\n")[1].replace("<|im_end|>","")
            user = ex["text"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            print(f"\n[{i+1}] {user[:70]}")
            print(asst[:400])
            print("---")

if __name__ == "__main__":
    main()