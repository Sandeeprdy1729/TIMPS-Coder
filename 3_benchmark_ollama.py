#!/usr/bin/env python3
"""
TIMPS-Coder — Ollama Benchmark Runner
======================================
Runs the same 25-test suite against the live Ollama model via HTTP API.
Saves results to benchmark_results.json and prints a scorecard.

Usage:
  python3 3_benchmark_ollama.py
  python3 3_benchmark_ollama.py --quick     # 10 tests only
  python3 3_benchmark_ollama.py --model sandeeprdy1729/timps-coder:latest
"""

import argparse
import json
import time
import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "sandeeprdy1729/timps-coder:latest"
OLLAMA_URL      = "http://localhost:11434/api/generate"
MAX_TOKENS      = 600
TEMP            = 0.1
RESULTS_FILE    = "benchmark_results.json"

SYSTEM = (
    "You are TIMPS-Coder v3, an elite bug-fixing and coding assistant built by "
    "Sandeep Reddy (TIMPS). For every task: THINK through the root cause, "
    "FIX with complete correct code, VERIFY edge cases."
)

TESTS = [
    # ── BUG ──────────────────────────────────────────────────────────────────
    {"id":"B01","cat":"BUG","label":"Spring @Autowired null",
     "input":"Fix: My Spring @Autowired service field is null inside the constructor. Why and how to fix?",
     "signals":["constructor","PostConstruct","@Autowired","because"],"need_code":True},
    {"id":"B02","cat":"BUG","label":"Dict KeyError nested access",
     "input":"Fix: `data['user']['email']` throws KeyError when email is sometimes absent.",
     "signals":[".get(","KeyError","None","default"],"need_code":True},
    {"id":"B03","cat":"BUG","label":"Off-by-one in for-loop",
     "input":"Fix: `for(int i=0; i<=arr.length; i++)` throws ArrayIndexOutOfBoundsException.",
     "signals":["<=","<","length","off-by-one"],"need_code":True},
    {"id":"B04","cat":"BUG","label":"Async/await fetch undefined",
     "input":"Fix: `const data = fetch(url); console.log(data.json)` gives undefined.",
     "signals":["await","async","Promise","then"],"need_code":True},
    {"id":"B05","cat":"BUG","label":"RecursionError no base case",
     "input":"Fix: `def fib(n): return fib(n-1) + fib(n-2)` hits RecursionError.",
     "signals":["base case","if n","n == 0","n <= 1"],"need_code":True},

    # ── SWE ──────────────────────────────────────────────────────────────────
    {"id":"S01","cat":"SWE","label":"ConcurrentModification iterator",
     "input":"Fix ConcurrentModificationException when deleting expired sessions inside a for-each loop over a HashMap.",
     "signals":["Iterator","removeIf","entrySet","ConcurrentHashMap"],"need_code":True},
    {"id":"S02","cat":"SWE","label":"Race condition in counter",
     "input":"Fix: two threads increment `self.count += 1` simultaneously causing wrong values.",
     "signals":["threading","Lock","atomic","race condition"],"need_code":True},
    {"id":"S03","cat":"SWE","label":"N+1 SQL queries",
     "input":"Fix: loading 100 products fires 101 SQL queries because each product triggers a separate category query.",
     "signals":["select_related","JOIN","prefetch_related","eager"],"need_code":True},
    {"id":"S04","cat":"SWE","label":"Memory leak event listener",
     "input":"Fix: browser memory grows because a React component attaches a resize listener but never removes it.",
     "signals":["removeEventListener","useEffect","cleanup","return"],"need_code":True},
    {"id":"S05","cat":"SWE","label":"Goroutine leak channel",
     "input":"Fix: goroutine count grows indefinitely; HTTP handler spawns a goroutine that blocks on a channel never closed.",
     "signals":["close(","context","defer","goroutine"],"need_code":True},

    # ── ALGO ─────────────────────────────────────────────────────────────────
    {"id":"A01","cat":"ALGO","label":"Two Sum O(n)",
     "input":"Solve Two Sum: given array and target, return indices of two numbers that add to target. O(n) required.",
     "signals":["dict","{}","complement","hash","O(n)"],"need_code":True},
    {"id":"A02","cat":"ALGO","label":"Sliding window max subarray",
     "input":"Find maximum sum subarray of size k using sliding window. Explain O(n) approach.",
     "signals":["window","sliding","O(n)","sum","max"],"need_code":True},
    {"id":"A03","cat":"ALGO","label":"Binary search rotated array",
     "input":"Search a target in rotated sorted array [4,5,6,7,0,1,2]. Return index or -1. O(log n).",
     "signals":["mid","left","right","log","binary"],"need_code":True},
    {"id":"A04","cat":"ALGO","label":"LRU Cache O(1)",
     "input":"Implement LRU Cache with get(key) and put(key, value) in O(1). Explain data structure.",
     "signals":["OrderedDict","deque","capacity","O(1)","evict"],"need_code":True},
    {"id":"A05","cat":"ALGO","label":"Merge K sorted lists",
     "input":"Merge k sorted linked lists into one sorted list. O(n log k) heap solution.",
     "signals":["heapq","heap","nsmallest","O(n log k)","priority"],"need_code":True},

    # ── CODE REVIEW ───────────────────────────────────────────────────────────
    {"id":"C01","cat":"CODE","label":"SQL injection",
     "input":"Review for security:\n```python\ndef get_user(name):\n    q = f\"SELECT * FROM users WHERE name='{name}'\"\n    return db.execute(q)\n```",
     "signals":["SQL injection","parameterized","?","sanitize","placeholder"],"need_code":True},
    {"id":"C02","cat":"CODE","label":"Nested loop O(n²)",
     "input":"Review and optimise:\n```python\ndef find_dupes(arr):\n    res=[]\n    for i in range(len(arr)):\n        for j in range(i+1,len(arr)):\n            if arr[i]==arr[j] and arr[i] not in res: res.append(arr[i])\n    return res\n```",
     "signals":["set","O(n)","O(n²)","dict","Counter"],"need_code":True},
    {"id":"C03","cat":"CODE","label":"Missing error handling async",
     "input":"Review for robustness:\n```js\nasync function loadUser(id){\n  const res=await fetch(`/api/users/${id}`);\n  return await res.json();\n}\n```",
     "signals":["try","catch","res.ok","status","error"],"need_code":True},
    {"id":"C04","cat":"CODE","label":"Mutable default argument",
     "input":"Find the bug:\n```python\ndef append_item(item, lst=[]):\n    lst.append(item)\n    return lst\n```",
     "signals":["mutable","default","None","shared","list"],"need_code":True},
    {"id":"C05","cat":"CODE","label":"String concat in loop Java",
     "input":"Review for performance:\n```java\nString r=\"\";\nfor(String s: items){ r+=s+\", \"; }\n```",
     "signals":["StringBuilder","append","O(n²)","immutable","performance"],"need_code":True},

    # ── AGENT ─────────────────────────────────────────────────────────────────
    {"id":"AG1","cat":"AGENT","label":"Debug failing test plan",
     "input":"pytest shows: FAILED test_auth.py::test_login - 401!=200 and test_logout - 500!=200. Give step-by-step debug plan.",
     "signals":["log","inspect","step","auth","status"],"need_code":False},
    {"id":"AG2","cat":"AGENT","label":"GitHub Actions CI pipeline",
     "input":"Write GitHub Actions YAML: run pytest on push, check with flake8, block merge if any test fails.",
     "signals":["on:","push:","jobs:","pytest","flake8"],"need_code":True},
    {"id":"AG3","cat":"AGENT","label":"Refactor monolith to package",
     "input":"Refactor 500-line monolithic app.py into proper package. What files to create, what to move, exact commands?",
     "signals":["mkdir","import","__init__","module","structure"],"need_code":True},
    {"id":"AG4","cat":"AGENT","label":"Fix flaky test timestamp",
     "input":"test_user_creation passes locally but fails 30% of time in CI checking timestamps. How to diagnose and fix?",
     "signals":["datetime","mock","freeze","timezone","race"],"need_code":True},
    {"id":"AG5","cat":"AGENT","label":"Profile slow API endpoint",
     "input":"/search API takes 4 seconds. Exact steps to profile, identify bottleneck, fix it.",
     "signals":["cProfile","time","profile","bottleneck","cache"],"need_code":False},
]

QUICK_IDS = {"B01","B02","B04","S01","A01","A02","C01","C02","AG1","AG2"}

COLORS = {
    "green":"\033[92m","red":"\033[91m","yellow":"\033[93m",
    "cyan":"\033[96m","bold":"\033[1m","reset":"\033[0m","dim":"\033[2m",
}

def c(color, text):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def query_ollama(model: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMP, "num_predict": MAX_TOKENS},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[OLLAMA_ERROR: {e}]"
    except Exception as e:
        return f"[ERROR: {e}]"


def build_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def score(output: str, test: dict) -> int:
    if not output or output.startswith("["):
        return 0
    has_code = (
        "```" in output
        or any(kw in output for kw in ["def ","class ","function ","const ","import ",
                                        "return ","if (","public ","jobs:","- name:"])
    )
    found = sum(1 for s in test["signals"] if s.lower() in output.lower())
    ratio = found / max(len(test["signals"]), 1)
    if test["need_code"] and not has_code:
        return 0
    if ratio >= 0.5 and (has_code or not test["need_code"]):
        return 2
    if ratio >= 0.25 or has_code:
        return 1
    return 0


def run_suite(model: str, tests: list) -> dict:
    results = {}
    for test in tests:
        t0 = time.time()
        output = query_ollama(model, build_prompt(test["input"]))
        elapsed = time.time() - t0
        pts = score(output, test)
        icon = c("green","✓") if pts==2 else c("yellow","~") if pts==1 else c("red","✗")
        print(f"  {icon}  [{test['id']}] {test['label']:<38} {pts}/2  ({elapsed:.1f}s)")
        results[test["id"]] = {"score": pts, "output": output[:600], "elapsed": round(elapsed, 2)}
    return results


def print_scorecard(results: dict, tests: list, model: str):
    cats = ["BUG","SWE","ALGO","CODE","AGENT"]
    cat_tests = {cat: [t for t in tests if t["cat"]==cat] for cat in cats}
    total_score = sum(v["score"] for v in results.values())
    total_max   = len(tests) * 2

    print(f"\n{'='*62}")
    print(f"  {c('bold','TIMPS-Coder Benchmark Scorecard')}")
    print(f"  Model: {c('cyan', model)}")
    print(f"{'='*62}")
    print(f"  {'Category':<22} {'Score':>8}   {'%':>6}")
    print(f"  {'─'*50}")

    for cat in cats:
        ts = cat_tests[cat]
        mx = len(ts) * 2
        sc = sum(results.get(t["id"],{}).get("score",0) for t in ts)
        pct = sc/mx*100 if mx>0 else 0
        bar_color = "green" if pct>=80 else "yellow" if pct>=50 else "red"
        label = {"BUG":"Bug Fix","SWE":"SWE / Repo-level","ALGO":"Algorithms",
                 "CODE":"Code Review","AGENT":"Agentic Reasoning"}[cat]
        print(f"  {label:<22} {c(bar_color, f'{sc}/{mx}'):>14}   {pct:>5.0f}%")

    print(f"  {'─'*50}")
    overall_pct = total_score / total_max * 100
    color = "green" if overall_pct>=75 else "yellow" if overall_pct>=50 else "red"
    print(f"  {'TOTAL':<22} {c(color, f'{total_score}/{total_max}'):>14}   {c(color, f'{overall_pct:.0f}%'):>6}\n")
    return total_score, total_max, overall_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    tests = [t for t in TESTS if t["id"] in QUICK_IDS] if args.quick else TESTS
    label = "quick (10)" if args.quick else "full (25)"

    print(f"\n  TIMPS-Coder Benchmark — {label}")
    print(f"  Model : {args.model}")
    print(f"  {'─'*58}")

    results = run_suite(args.model, tests)
    total_score, total_max, pct = print_scorecard(results, tests, args.model)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": args.model, "tests": len(tests),
                   "total_score": total_score, "total_max": total_max,
                   "overall_pct": round(pct, 1),
                   "results": results}, f, indent=2)

    print(f"  Results saved → {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
