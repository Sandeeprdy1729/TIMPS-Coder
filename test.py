"""
TIMPS-Coder — Complete Test Suite
===================================
Run: python3 test.py

Tests your model on 10 real bugs across Java, Python, JavaScript.
Compares against the base model and prints a score card.
"""

import subprocess, sys, time, json
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TUNED_MODEL = "./timps-coder-fused"
MAX_TOKENS  = 400
TEMP        = 0.1
# ────────────────────────────────────────────────────────────

SYSTEM = """You are TIMPS-Coder, an expert coding assistant built by Sandeep Reddy. When fixing a bug:
1. Explain WHY the bug occurs in plain English
2. Show the corrected code with proper formatting"""

TESTS = [
    {
        "id": "T01", "lang": "Java", "category": "NullPointer",
        "input": "Fix null_pointer: Why is my Spring @Autowired field null?",
        "must_contain": ["because", "new ", "Spring"],
        "must_have_code": True,
    },
    {
        "id": "T02", "lang": "Python", "category": "KeyError",
        "input": "Fix key_error: Python dict raises KeyError on data['user']['email'] — the key sometimes doesn't exist.",
        "must_contain": [".get(", "KeyError", "None"],
        "must_have_code": True,
    },
    {
        "id": "T03", "lang": "Java", "category": "IndexOutOfBounds",
        "input": "Fix index_error: Java for loop `for(int i=0; i<=arr.length; i++)` throws ArrayIndexOutOfBoundsException.",
        "must_contain": ["<=", "<", "length"],
        "must_have_code": True,
    },
    {
        "id": "T04", "lang": "Python", "category": "TypeError",
        "input": "Fix type_error: `total = count + user_input` raises TypeError int + str. user_input comes from input().",
        "must_contain": ["int(", "str", "convert"],
        "must_have_code": True,
    },
    {
        "id": "T05", "lang": "JavaScript", "category": "AsyncBug",
        "input": "Fix async_bug: fetch() returns a Promise, not data. `const data = fetch(url); console.log(data.json)` gives undefined.",
        "must_contain": ["await", "async", "Promise"],
        "must_have_code": True,
    },
    {
        "id": "T06", "lang": "Java", "category": "NullPointer",
        "input": "Fix null_pointer: Collectors.toMap throws NullPointerException when one of the values is null.",
        "must_contain": ["HashMap", "null", "merge"],
        "must_have_code": True,
    },
    {
        "id": "T07", "lang": "JavaScript", "category": "ScopeBug",
        "input": "Fix scope_bug: `for(var i=0; i<3; i++) { setTimeout(()=>console.log(i), 100) }` prints 3,3,3 instead of 0,1,2.",
        "must_contain": ["let", "closure", "var"],
        "must_have_code": True,
    },
    {
        "id": "T08", "lang": "Java", "category": "ConcurrentModification",
        "input": "Fix concurrent_modification: ConcurrentModificationException when removing from a List inside a for-each loop.",
        "must_contain": ["Iterator", "removeIf", "iterator"],
        "must_have_code": True,
    },
    {
        "id": "T09", "lang": "Python", "category": "RecursionError",
        "input": "Fix recursion_error: `def fib(n): return fib(n-1) + fib(n-2)` hits RecursionError. No base case.",
        "must_contain": ["base case", "if n", "n == 0"],
        "must_have_code": True,
    },
    {
        "id": "T10", "lang": "Python", "category": "LogicError",
        "input": "Fix logic_error: Function returns second largest from list but returns the largest instead. Code: sorted(nums)[-2]",
        "must_contain": ["set(", "unique", "duplicate"],
        "must_have_code": True,
    },
]


def build_prompt(instruction):
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run(model, prompt):
    start = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "mlx_lm.generate",
         "--model", model, "--max-tokens", str(MAX_TOKENS),
         "--temp", str(TEMP), "--prompt", prompt],
        capture_output=True, text=True
    )
    elapsed = time.time() - start
    out = r.stdout.strip()
    if "<|im_start|>assistant" in out:
        out = out.split("<|im_start|>assistant")[-1].strip()
    out = out.replace("<|im_end|>", "").strip()
    return out, round(elapsed, 1)


def score(output, must_contain, must_have_code):
    """
    Score 0–5:
      +1  Has explanation (>80 chars of prose)
      +1  Has code block
      +1  At least one expected keyword present
      +1  Response >150 chars (not a cop-out)
      +1  No repetition (degenerate loop)
    """
    s = 0
    import re

    prose = re.sub(r"```.*?```", "", output, flags=re.DOTALL).strip()
    has_code = "```" in output
    has_prose = len(prose) > 80
    has_keyword = any(k.lower() in output.lower() for k in must_contain)
    long_enough = len(output) > 150

    # Detect repetition: same line 3+ times
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    from collections import Counter
    counts = Counter(lines)
    no_repeat = max(counts.values(), default=0) < 3

    if has_prose:    s += 1
    if has_code:     s += 1
    if has_keyword:  s += 1
    if long_enough:  s += 1
    if no_repeat:    s += 1

    return s, {
        "has_prose": has_prose,
        "has_code": has_code,
        "has_keyword": has_keyword,
        "long_enough": long_enough,
        "no_repeat": no_repeat,
    }


def run_tests(model_path, label):
    print(f"\n{'='*55}")
    print(f"  Testing: {label}")
    print(f"{'='*55}")
    results = []
    for t in TESTS:
        prompt = build_prompt(t["input"])
        output, elapsed = run(model_path, prompt)
        s, breakdown = score(output, t["must_contain"], t["must_have_code"])
        
        checks = " ".join(
            ("✅" if v else "❌") + k[:5]
            for k, v in breakdown.items()
        )
        print(f"[{t['id']}] {t['lang']}/{t['category']:<25} {s}/5  {checks}  ({elapsed}s)")
        if s < 3:
            print(f"       Output: {output[:120]}...")
        results.append({"id": t["id"], "score": s, "output": output, "time": elapsed})
    return results


def main():
    print("\n🧪 TIMPS-Coder Test Suite")
    print("="*55)

    base_results  = run_tests(BASE_MODEL,  "BASE  — Qwen2.5-Coder-0.5B (untrained)")
    tuned_results = run_tests(TUNED_MODEL, "TUNED — TIMPS-Coder (your model)")

    # Summary
    base_total  = sum(r["score"] for r in base_results)
    tuned_total = sum(r["score"] for r in tuned_results)
    max_score   = len(TESTS) * 5

    wins   = sum(1 for b, t in zip(base_results, tuned_results) if t["score"] > b["score"])
    ties   = sum(1 for b, t in zip(base_results, tuned_results) if t["score"] == b["score"])
    losses = sum(1 for b, t in zip(base_results, tuned_results) if t["score"] < b["score"])

    print(f"\n{'='*55}")
    print(f"  FINAL SCORE CARD")
    print(f"{'='*55}")
    print(f"  Base model:   {base_total}/{max_score}  ({100*base_total/max_score:.0f}%)")
    print(f"  TIMPS-Coder:  {tuned_total}/{max_score}  ({100*tuned_total/max_score:.0f}%)")
    print(f"  Improvement:  +{tuned_total - base_total} points")
    print(f"  Win/Tie/Loss: {wins}W / {ties}T / {losses}L out of {len(TESTS)}")
    print(f"{'='*55}")

    if tuned_total > base_total:
        pct_gain = 100 * (tuned_total - base_total) / max(base_total, 1)
        print(f"\n🚀 TIMPS-Coder is {pct_gain:.0f}% better than base!")
        print("   Ready to publish. Run: python3 publish.py")
    elif tuned_total == base_total:
        print("\n⚠️  No improvement. Check training logs — may need more iters.")
    else:
        print("\n❌ Regression. Dataset or training issue. Check train.jsonl quality.")

    # Save results
    with open("test_results.json", "w") as f:
        json.dump({
            "base":  {"total": base_total,  "results": base_results},
            "tuned": {"total": tuned_total, "results": tuned_results},
            "summary": {
                "base_pct": round(100*base_total/max_score, 1),
                "tuned_pct": round(100*tuned_total/max_score, 1),
                "wins": wins, "ties": ties, "losses": losses,
            }
        }, f, indent=2)
    print("\n  📄 Full results saved → test_results.json")
    print("  (Use these numbers in your HuggingFace model card)")


if __name__ == "__main__":
    main()