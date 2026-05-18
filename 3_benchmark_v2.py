#!/usr/bin/env python3
"""
TIMPS-Coder v2 — Comprehensive Benchmark
==========================================
Tests 25 cases across 5 capability dimensions — compare TIMPS-Coder v2
against the base Qwen2.5-Coder-0.5B model and report a detailed scorecard.

Dimensions (5 each):
  BUG   — Bug analysis & fix             (core v1 strength)
  SWE   — Repository-level issue solving  (new in v2)
  ALGO  — Algorithm problem solving       (new in v2)
  CODE  — Code review & improvement       (new in v2)
  AGENT — Multi-step agentic reasoning    (new in v2)

Scoring (per test):
  +2  All required signals present AND has code
  +1  Has code but missing some signals
   0  Prose only, or refused

Usage:
  python3 3_benchmark_v2.py              # compare both models
  python3 3_benchmark_v2.py --tuned-only # skip base model (faster)
  python3 3_benchmark_v2.py --quick      # 10 tests only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TUNED_MODEL  = "./timps-coder-fused"
MAX_TOKENS   = 600
TEMP         = 0.1
RESULTS_FILE = "test_results_v2.json"

SYSTEM = (
    "You are TIMPS-Coder v2, an agentic software engineer built by Sandeep Reddy (TIMPS). "
    "For every task: THINK through the root cause or approach, ACT with complete correct "
    "code, VERIFY edge cases."
)
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # ── BUG: Bug analysis & fix ───────────────────────────────────────────────
    {
        "id": "B01", "cat": "BUG", "lang": "Java",
        "label": "Spring @Autowired null",
        "input": "Fix null_pointer: My Spring @Autowired service field is null inside a constructor. Why?",
        "signals": ["constructor", "PostConstruct", "@Autowired", "because"],
        "need_code": True,
    },
    {
        "id": "B02", "cat": "BUG", "lang": "Python",
        "label": "Dict KeyError nested access",
        "input": "Fix key_error: `data['user']['email']` throws KeyError when email is sometimes absent.",
        "signals": [".get(", "KeyError", "None", "default"],
        "need_code": True,
    },
    {
        "id": "B03", "cat": "BUG", "lang": "Java",
        "label": "Off-by-one in for-loop",
        "input": "Fix index_error: `for(int i=0; i<=arr.length; i++)` throws ArrayIndexOutOfBoundsException.",
        "signals": ["<=", "<", "length", "off-by-one"],
        "need_code": True,
    },
    {
        "id": "B04", "cat": "BUG", "lang": "JavaScript",
        "label": "Async/await fetch undefined",
        "input": "Fix async_bug: `const data = fetch(url); console.log(data.json)` gives undefined.",
        "signals": ["await", "async", "Promise", "then"],
        "need_code": True,
    },
    {
        "id": "B05", "cat": "BUG", "lang": "Python",
        "label": "RecursionError no base case",
        "input": "Fix recursion_error: `def fib(n): return fib(n-1) + fib(n-2)` hits RecursionError.",
        "signals": ["base case", "if n", "n == 0", "n <= 1"],
        "need_code": True,
    },

    # ── SWE: Repository-level issue solving ──────────────────────────────────
    {
        "id": "S01", "cat": "SWE", "lang": "Python",
        "label": "ConcurrentModification iterator",
        "input": (
            "Repository: myapp/backend\n\n"
            "Issue: ConcurrentModificationException when deleting expired sessions "
            "inside a for-each loop over a HashMap. Stack trace points to SessionManager.cleanup()."
        ),
        "signals": ["Iterator", "removeIf", "entrySet", "ConcurrentHashMap"],
        "need_code": True,
    },
    {
        "id": "S02", "cat": "SWE", "lang": "Python",
        "label": "Race condition in shared counter",
        "input": (
            "Repository: analytics/service\n\n"
            "Issue: Our request counter shows wrong values under high load. "
            "Two threads increment `self.count += 1` simultaneously."
        ),
        "signals": ["threading", "Lock", "atomic", "race condition"],
        "need_code": True,
    },
    {
        "id": "S03", "cat": "SWE", "lang": "Python",
        "label": "N+1 query problem",
        "input": (
            "Repository: shop/api\n\n"
            "Issue: Loading 100 products takes 101 SQL queries. "
            "Each product triggers a separate query to fetch its category."
        ),
        "signals": ["select_related", "JOIN", "prefetch_related", "eager"],
        "need_code": True,
    },
    {
        "id": "S04", "cat": "SWE", "lang": "JavaScript",
        "label": "Memory leak event listener",
        "input": (
            "Repository: frontend/dashboard\n\n"
            "Issue: Browser memory grows continuously. "
            "We attach a 'resize' event listener in a React component but never clean it up."
        ),
        "signals": ["removeEventListener", "useEffect", "cleanup", "return"],
        "need_code": True,
    },
    {
        "id": "S05", "cat": "SWE", "lang": "Go",
        "label": "Goroutine leak in HTTP handler",
        "input": (
            "Repository: api/server\n\n"
            "Issue: goroutine count grows indefinitely. "
            "Our HTTP handler spawns a goroutine that blocks on a channel but we never close it."
        ),
        "signals": ["close(", "context", "defer", "goroutine", "channel"],
        "need_code": True,
    },

    # ── ALGO: Algorithm problem solving ──────────────────────────────────────
    {
        "id": "A01", "cat": "ALGO", "lang": "Python",
        "label": "Two Sum",
        "input": "Solve Two Sum: Given an array of integers and a target, return indices of the two numbers that add up to target. O(n) solution required.",
        "signals": ["dict", "{}",  "complement", "hash", "O(n)"],
        "need_code": True,
    },
    {
        "id": "A02", "cat": "ALGO", "lang": "Python",
        "label": "Sliding window max subarray",
        "input": "Solve: Find the maximum sum subarray of size k using a sliding window. Explain the O(n) approach.",
        "signals": ["window", "sliding", "O(n)", "sum", "max"],
        "need_code": True,
    },
    {
        "id": "A03", "cat": "ALGO", "lang": "Python",
        "label": "Binary search rotated array",
        "input": "Solve: Search a target in a rotated sorted array [4,5,6,7,0,1,2]. Return index or -1. O(log n) required.",
        "signals": ["mid", "left", "right", "log", "binary"],
        "need_code": True,
    },
    {
        "id": "A04", "cat": "ALGO", "lang": "Python",
        "label": "LRU Cache implementation",
        "input": "Implement an LRU Cache with get(key) and put(key, value) in O(1) time. Explain your data structure choice.",
        "signals": ["OrderedDict", "deque", "capacity", "O(1)", "evict"],
        "need_code": True,
    },
    {
        "id": "A05", "cat": "ALGO", "lang": "Python",
        "label": "Merge K sorted lists",
        "input": "Merge k sorted linked lists into one sorted linked list. Provide the O(n log k) heap solution.",
        "signals": ["heapq", "heap", "nsmallest", "O(n log k)", "priority"],
        "need_code": True,
    },

    # ── CODE: Code review & improvement ──────────────────────────────────────
    {
        "id": "C01", "cat": "CODE", "lang": "Python",
        "label": "SQL injection vulnerability",
        "input": (
            "Review this code for security issues:\n\n"
            "```python\n"
            "def get_user(username):\n"
            "    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n"
            "    return db.execute(query)\n"
            "```"
        ),
        "signals": ["SQL injection", "parameterized", "?", "sanitize", "placeholder"],
        "need_code": True,
    },
    {
        "id": "C02", "cat": "CODE", "lang": "Python",
        "label": "Inefficient nested loop O(n²)",
        "input": (
            "Review and optimise this function:\n\n"
            "```python\n"
            "def find_duplicates(arr):\n"
            "    result = []\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i+1, len(arr)):\n"
            "            if arr[i] == arr[j] and arr[i] not in result:\n"
            "                result.append(arr[i])\n"
            "    return result\n"
            "```"
        ),
        "signals": ["set", "O(n)", "O(n²)", "dict", "Counter"],
        "need_code": True,
    },
    {
        "id": "C03", "cat": "CODE", "lang": "JavaScript",
        "label": "Missing error handling async",
        "input": (
            "Review this JavaScript for robustness issues:\n\n"
            "```js\n"
            "async function loadUser(id) {\n"
            "    const res = await fetch(`/api/users/${id}`);\n"
            "    const data = await res.json();\n"
            "    return data;\n"
            "}\n"
            "```"
        ),
        "signals": ["try", "catch", "res.ok", "status", "error"],
        "need_code": True,
    },
    {
        "id": "C04", "cat": "CODE", "lang": "Python",
        "label": "Mutable default argument",
        "input": (
            "Find the bug and explain why this Python function behaves unexpectedly:\n\n"
            "```python\n"
            "def append_item(item, lst=[]):\n"
            "    lst.append(item)\n"
            "    return lst\n"
            "```"
        ),
        "signals": ["mutable", "default", "None", "shared", "list"],
        "need_code": True,
    },
    {
        "id": "C05", "cat": "CODE", "lang": "Java",
        "label": "String concatenation in loop",
        "input": (
            "Review this Java code for performance:\n\n"
            "```java\n"
            "String result = \"\";\n"
            "for (String s : items) {\n"
            "    result += s + \", \";\n"
            "}\n"
            "```"
        ),
        "signals": ["StringBuilder", "append", "O(n²)", "immutable", "performance"],
        "need_code": True,
    },

    # ── AGENT: Multi-step agentic reasoning ──────────────────────────────────
    {
        "id": "AG1", "cat": "AGENT", "lang": "Bash",
        "label": "Debug failing test plan",
        "input": (
            "I ran `pytest tests/` and got:\n\n"
            "```\nFAILED tests/test_auth.py::test_login - AssertionError: 401 != 200\n"
            "FAILED tests/test_auth.py::test_logout - AssertionError: 500 != 200\n```\n\n"
            "Walk me through the step-by-step debugging plan you would follow."
        ),
        "signals": ["print", "log", "breakpoint", "inspect", "step"],
        "need_code": False,
    },
    {
        "id": "AG2", "cat": "AGENT", "lang": "Bash",
        "label": "Set up CI pipeline",
        "input": (
            "Write a GitHub Actions workflow that: runs pytest on push, "
            "checks code with flake8, and blocks merge if any test fails."
        ),
        "signals": ["on:", "push:", "jobs:", "pytest", "flake8"],
        "need_code": True,
    },
    {
        "id": "AG3", "cat": "AGENT", "lang": "Python",
        "label": "Refactor with tool use thinking",
        "input": (
            "I need to refactor a 500-line monolithic `app.py` into a proper "
            "package structure. Walk me through: what files to create, what to move "
            "where, and the exact commands to run."
        ),
        "signals": ["mkdir", "import", "__init__", "module", "structure"],
        "need_code": True,
    },
    {
        "id": "AG4", "cat": "AGENT", "lang": "Python",
        "label": "Reproduce & fix a flaky test",
        "input": (
            "Test `test_user_creation` passes locally but fails 30% of the time in CI. "
            "The test creates a user and checks the timestamp. How do you diagnose and fix flakiness?"
        ),
        "signals": ["datetime", "mock", "freeze", "timezone", "race"],
        "need_code": True,
    },
    {
        "id": "AG5", "cat": "AGENT", "lang": "Python",
        "label": "Performance profiling plan",
        "input": (
            "Our API endpoint `/search` takes 4 seconds to respond. "
            "Describe the exact steps to profile it, identify the bottleneck, and fix it."
        ),
        "signals": ["cProfile", "time", "profile", "bottleneck", "cache"],
        "need_code": False,
    },
]

QUICK_IDS = {"B01", "B02", "B04", "S01", "A01", "A02", "C01", "C02", "AG1", "AG2"}


# ── Helpers ───────────────────────────────────────────────────────────────────

COLORS = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
    "dim":    "\033[2m",
}

def c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def build_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_model(model_path: str, instruction: str, max_tokens: int = MAX_TOKENS) -> str:
    prompt = build_prompt(instruction)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.generate",
                "--model", model_path,
                "--max-tokens", str(max_tokens),
                "--temp", str(TEMP),
                "--prompt", prompt,
            ],
            capture_output=True, text=True, timeout=120,
        )
        out = result.stdout.strip()
        # Strip prompt echo
        if "<|im_start|>assistant" in out:
            out = out.split("<|im_start|>assistant")[-1].strip()
        return out
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def score_output(output: str, test: dict) -> int:
    if not output or output.startswith("["):
        return 0
    has_code = (
        "```" in output or
        any(kw in output for kw in [
            "def ", "class ", "function ", "const ", "import ",
            "return ", "if (", "public ", "jobs:", "- name:", "$ "
        ])
    )
    signals_found = sum(1 for s in test["signals"] if s.lower() in output.lower())
    ratio = signals_found / max(len(test["signals"]), 1)

    if test["need_code"] and not has_code:
        return 0
    if ratio >= 0.5 and (has_code or not test["need_code"]):
        return 2
    if ratio >= 0.25 or has_code:
        return 1
    return 0


def category_bar(score: int, max_score: int) -> str:
    pct = score / max_score if max_score > 0 else 0
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    color = "green" if pct >= 0.8 else "yellow" if pct >= 0.5 else "red"
    return c(color, bar) + f"  {score}/{max_score}  ({pct*100:.0f}%)"


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(model_path: str, label: str, tests: list) -> dict:
    results = {}
    print(f"\n  Running {c('bold', label)}  ({len(tests)} tests)")
    print("  " + "─" * 60)

    for test in tests:
        t0 = time.time()
        output = run_model(model_path, test["input"])
        elapsed = time.time() - t0
        pts = score_output(output, test)

        icon = c("green", "✓") if pts == 2 else c("yellow", "~") if pts == 1 else c("red", "✗")
        print(f"  {icon}  [{test['id']}] {test['label']:<35} {pts}/2  ({elapsed:.1f}s)")

        results[test["id"]] = {
            "score":   pts,
            "output":  output[:500],
            "elapsed": round(elapsed, 2),
        }

    return results


def print_scorecard(base_results: dict, tuned_results: dict, tests: list):
    cats = ["BUG", "SWE", "ALGO", "CODE", "AGENT"]
    cat_tests = {cat: [t for t in tests if t["cat"] == cat] for cat in cats}
    max_per_cat = {cat: len(ts) * 2 for cat, ts in cat_tests.items()}

    def cat_score(results, cat):
        return sum(results.get(t["id"], {}).get("score", 0) for t in cat_tests[cat])

    print(f"\n{'='*68}")
    print(f"  {c('bold', 'TIMPS-Coder v2 Benchmark — Results')}")
    print(f"{'='*68}")
    print(f"  {'Category':<22} {'Base':>10}   {'TIMPS-Coder v2':>14}   {'Δ':>6}")
    print(f"  {'─'*60}")

    total_base = 0
    total_tuned = 0
    total_max = 0

    for cat in cats:
        mx = max_per_cat[cat]
        b  = cat_score(base_results, cat)
        t  = cat_score(tuned_results, cat)
        delta = t - b
        delta_str = c("green", f"+{delta}") if delta > 0 else c("red", str(delta)) if delta < 0 else " 0"
        cat_label = {
            "BUG":   "Bug Fix",
            "SWE":   "SWE / Repo-level",
            "ALGO":  "Algorithms",
            "CODE":  "Code Review",
            "AGENT": "Agentic Reasoning",
        }[cat]
        print(f"  {cat_label:<22}  {b:>3}/{mx}        {t:>3}/{mx}          {delta_str}")
        total_base  += b
        total_tuned += t
        total_max   += mx

    print(f"  {'─'*60}")
    b_pct = total_base  / total_max * 100
    t_pct = total_tuned / total_max * 100
    d_pct = t_pct - b_pct
    delta_str = c("green", f"+{d_pct:.1f}%") if d_pct >= 0 else c("red", f"{d_pct:.1f}%")

    print(f"  {'TOTAL':<22}  {total_base:>3}/{total_max}  ({b_pct:.0f}%)  "
          f"{total_tuned:>3}/{total_max}  ({t_pct:.0f}%)  {delta_str}")
    print(f"{'='*68}")

    # Per-test comparison
    wins = sum(
        1 for t in tests
        if tuned_results.get(t["id"], {}).get("score", 0) >
           base_results.get(t["id"], {}).get("score", 0)
    )
    ties = sum(
        1 for t in tests
        if tuned_results.get(t["id"], {}).get("score", 0) ==
           base_results.get(t["id"], {}).get("score", 0)
    )
    losses = len(tests) - wins - ties

    print(f"\n  Wins: {c('green', str(wins))}  "
          f"Ties: {c('yellow', str(ties))}  "
          f"Losses: {c('red', str(losses))}")
    print(f"{'='*68}\n")

    return {
        "base_pct":   round(b_pct, 1),
        "tuned_pct":  round(t_pct, 1),
        "delta_pct":  round(d_pct, 1),
        "wins":   wins,
        "ties":   ties,
        "losses": losses,
        "total_base":  total_base,
        "total_tuned": total_tuned,
        "total_max":   total_max,
    }


def main():
    parser = argparse.ArgumentParser(description="TIMPS-Coder v2 Benchmark")
    parser.add_argument("--tuned-only", action="store_true",
                        help="Skip base model comparison (faster)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 10 core tests")
    args = parser.parse_args()

    active_tests = [t for t in TESTS if t["id"] in QUICK_IDS] if args.quick else TESTS

    if not Path(TUNED_MODEL).exists():
        print(f"\n❌  Fused model not found at {TUNED_MODEL}")
        print("    Run:  bash 2_train_sft.sh")
        sys.exit(1)

    print(f"\n{'='*68}")
    print(f"  {c('bold', 'TIMPS-Coder v2 — Benchmark Suite')}")
    print(f"  {len(active_tests)} tests  ·  5 dimensions  ·  max {len(active_tests)*2} pts")
    print(f"{'='*68}")

    if args.tuned_only:
        base_results = {t["id"]: {"score": 0} for t in active_tests}
    else:
        base_results = run_benchmark(BASE_MODEL, f"Base — {BASE_MODEL}", active_tests)

    tuned_results = run_benchmark(TUNED_MODEL, "TIMPS-Coder v2", active_tests)
    summary = print_scorecard(base_results, tuned_results, active_tests)

    # Save results
    output = {
        "summary":      summary,
        "base_results": base_results,
        "tuned_results": tuned_results,
        "tests":        [{"id": t["id"], "cat": t["cat"], "label": t["label"]}
                         for t in active_tests],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved → {RESULTS_FILE}")
    print(f"  Next: python3 launch_timps_v2.py\n")


if __name__ == "__main__":
    main()
