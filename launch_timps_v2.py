#!/usr/bin/env python3
"""
TIMPS-Coder v2 — Launch & Showcase
=====================================
Demonstrates what makes TIMPS-Coder v2 different from every other
0.5B coding model. Five capability chapters, each with a live demo.

Chapters:
  1. Deep Bug Analysis  — root-cause reasoning, not just pattern matching
  2. SWE / Repo-Level   — GitHub-issue-style multi-file understanding
  3. Algorithm Mastery  — competitive programming with complexity analysis
  4. Code Review Agent  — security + performance + correctness in one pass
  5. Agentic Planning   — multi-step task planning like a senior engineer

Modes:
  python3 launch_timps_v2.py            # full showcase (all 5 chapters)
  python3 launch_timps_v2.py --chat     # interactive REPL
  python3 launch_timps_v2.py --chapter 2   # single chapter demo
  python3 launch_timps_v2.py --quick    # 1 demo per chapter
"""

import argparse
import subprocess
import sys
import time
import textwrap
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = "./adapters"
MAX_TOKENS  = 700
TEMP        = 0.1

SYSTEM = (
    "You are TIMPS-Coder v2, an agentic software engineer built by Sandeep Reddy (TIMPS). "
    "For every task: THINK through the root cause or approach, ACT with complete correct "
    "code, VERIFY edge cases and complexity."
)

BANNER = r"""
 ████████╗██╗███╗   ███╗██████╗ ███████╗       ██████╗ ██████╗ ██████╗ ███████╗██████╗
    ██╔══╝██║████╗ ████║██╔══██╗██╔════╝      ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
    ██║   ██║██╔████╔██║██████╔╝███████╗      ██║     ██║   ██║██║  ██║█████╗  ██████╔╝
    ██║   ██║██║╚██╔╝██║██╔═══╝ ╚════██║      ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
    ██║   ██║██║ ╚═╝ ██║██║     ███████║      ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
    ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝       ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
"""

# ── Color helpers ─────────────────────────────────────────────────────────────
def c(code, text):
    codes = {
        "bold":   "\033[1m",   "dim":    "\033[2m",
        "green":  "\033[92m",  "red":    "\033[91m",
        "yellow": "\033[93m",  "blue":   "\033[94m",
        "cyan":   "\033[96m",  "white":  "\033[97m",
        "purple": "\033[95m",  "reset":  "\033[0m",
    }
    return f"{codes.get(code,'')}{text}\033[0m"

def hr(ch="═", width=72):
    return ch * width

def section(title: str, icon: str = "◆"):
    print(f"\n{c('cyan', hr())}")
    print(f"  {icon}  {c('bold', title)}")
    print(c("cyan", hr()))

def demo_header(label: str, lang: str):
    print(f"\n  {c('yellow', '▶')}  {c('bold', label)}  {c('dim', f'[{lang}]')}")
    print(f"  {c('dim', '─' * 60)}")


# ── Showcases ─────────────────────────────────────────────────────────────────

CHAPTERS = [
    {
        "num": 1,
        "title": "Deep Bug Analysis & Fix",
        "subtitle": "Root-cause reasoning — not just pattern matching",
        "icon": "🐛",
        "demos": [
            {
                "label": "Spring @Autowired field null inside constructor",
                "lang": "Java",
                "prompt": (
                    "Fix null_pointer: My Spring @Autowired UserService is null "
                    "when I call it from the constructor of OrderController. "
                    "Why does Spring fail here and what's the correct pattern?"
                ),
            },
            {
                "label": "JavaScript closure captures wrong loop variable",
                "lang": "JavaScript",
                "prompt": (
                    "Fix scope_bug: This prints '3,3,3' instead of '0,1,2':\n\n"
                    "```js\n"
                    "for (var i = 0; i < 3; i++) {\n"
                    "  setTimeout(() => console.log(i), 100);\n"
                    "}\n"
                    "```\n\n"
                    "Explain the root cause and show all three fix strategies."
                ),
            },
        ],
    },
    {
        "num": 2,
        "title": "SWE / Repository-Level Reasoning",
        "subtitle": "Think like a senior engineer navigating a real codebase",
        "icon": "🔧",
        "demos": [
            {
                "label": "N+1 query problem in Django ORM",
                "lang": "Python/Django",
                "prompt": (
                    "Repository: ecommerce/api\n\n"
                    "Issue: The /products endpoint hits the database 101 times "
                    "when fetching 100 products. Each product triggers a separate "
                    "query for its category.\n\n"
                    "Diagnose the N+1 problem and show the exact ORM fix with "
                    "`select_related` or `prefetch_related`. Include a before/after "
                    "comparison and explain when to use each."
                ),
            },
            {
                "label": "Thread-safety race condition in shared counter",
                "lang": "Python",
                "prompt": (
                    "Repository: analytics/tracker\n\n"
                    "Issue: Under high load our request counter shows incorrect values. "
                    "Two threads run `self.count += 1` simultaneously.\n\n"
                    "Provide a thread-safe implementation using the right Python "
                    "primitive. Show both the broken version and the fixed version."
                ),
            },
        ],
    },
    {
        "num": 3,
        "title": "Algorithm Mastery",
        "subtitle": "Competitive-level solutions with explicit complexity analysis",
        "icon": "⚡",
        "demos": [
            {
                "label": "LRU Cache — O(1) get & put",
                "lang": "Python",
                "prompt": (
                    "Implement an LRU Cache that supports:\n"
                    "  - `get(key)` → value or -1 if absent   O(1)\n"
                    "  - `put(key, value)` → evicts LRU if at capacity   O(1)\n\n"
                    "Explain your data structure choice, then provide the full "
                    "implementation with clear comments."
                ),
            },
            {
                "label": "Sliding window — Maximum sum subarray of size k",
                "lang": "Python",
                "prompt": (
                    "Given an array of integers and a number k, find the maximum "
                    "sum of any contiguous subarray of size k in O(n) time.\n\n"
                    "Walk through the sliding window intuition, implement it, "
                    "and analyse time and space complexity."
                ),
            },
        ],
    },
    {
        "num": 4,
        "title": "Code Review Agent",
        "subtitle": "Finds security, performance, and correctness issues simultaneously",
        "icon": "🔍",
        "demos": [
            {
                "label": "SQL injection + missing error handling",
                "lang": "Python",
                "prompt": (
                    "Review this authentication handler and identify ALL issues:\n\n"
                    "```python\n"
                    "def login(username, password):\n"
                    "    query = f\"SELECT * FROM users WHERE name='{username}' "
                    "AND pass='{password}'\"\n"
                    "    user = db.execute(query).fetchone()\n"
                    "    if user:\n"
                    "        session['user'] = user\n"
                    "        return redirect('/dashboard')\n"
                    "```\n\n"
                    "List every issue (security, correctness, missing checks) "
                    "then provide the fully corrected version."
                ),
            },
            {
                "label": "React memory leak + missing dependency array",
                "lang": "JavaScript/React",
                "prompt": (
                    "Review this React component for all issues:\n\n"
                    "```jsx\n"
                    "function Dashboard({ userId }) {\n"
                    "  const [data, setData] = useState(null);\n\n"
                    "  useEffect(() => {\n"
                    "    fetch(`/api/user/${userId}`)\n"
                    "      .then(r => r.json())\n"
                    "      .then(d => setData(d));\n"
                    "    window.addEventListener('resize', handleResize);\n"
                    "  }, []);\n\n"
                    "  return <div>{data?.name}</div>;\n"
                    "}\n"
                    "```\n\n"
                    "Find every bug and rewrite the component correctly."
                ),
            },
        ],
    },
    {
        "num": 5,
        "title": "Agentic Task Planning",
        "subtitle": "Multi-step thinking — plans + commands like a senior engineer",
        "icon": "🤖",
        "demos": [
            {
                "label": "Debug failing CI — systematic diagnosis plan",
                "lang": "Bash/Python",
                "prompt": (
                    "Our CI pipeline is failing with:\n\n"
                    "```\n"
                    "FAILED tests/test_auth.py::test_login - AssertionError: 401 != 200\n"
                    "FAILED tests/test_payment.py::test_charge - ConnectionError\n"
                    "```\n\n"
                    "Give me the exact step-by-step debugging plan including:\n"
                    "1. The terminal commands to run first\n"
                    "2. How to isolate whether it's a config vs code issue\n"
                    "3. How to reproduce locally\n"
                    "4. What to check in the test setup"
                ),
            },
            {
                "label": "Refactor monolith into package structure",
                "lang": "Python/Bash",
                "prompt": (
                    "I have a 600-line `app.py` that needs to be refactored into a "
                    "proper Python package. It contains: routes, database models, "
                    "authentication logic, and utility helpers.\n\n"
                    "Provide:\n"
                    "1. The target directory structure\n"
                    "2. What goes in each file\n"
                    "3. The exact shell commands to create it\n"
                    "4. How to update imports without breaking anything"
                ),
            },
        ],
    },
]


# ── Core runner ───────────────────────────────────────────────────────────────

def run_model(prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[str, float]:
    full_prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    t0 = time.time()
    try:
        cmd = [
            sys.executable, "-m", "mlx_lm.generate",
            "--model", BASE_MODEL,
            "--adapter-path", ADAPTER_PATH,
            "--max-tokens", str(max_tokens),
            "--temp", str(TEMP),
            "--prompt", full_prompt,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        out = result.stdout.strip()
        if "<|im_start|>assistant" in out:
            out = out.split("<|im_start|>assistant")[-1].strip()
        # Strip trailing stop tokens
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            if tok in out:
                out = out[:out.index(tok)].strip()
        return out, round(time.time() - t0, 1)
    except subprocess.TimeoutExpired:
        return "[TIMEOUT — model took >3 min]", 180.0
    except Exception as e:
        return f"[ERROR: {e}]", 0.0


def show_output(text: str, elapsed: float):
    """Pretty-print model output with word-wrap."""
    lines = text.split("\n")
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            print(c("dim", f"  {line}"))
            continue
        if in_code:
            print(c("cyan", f"  {line}"))
        else:
            wrapped = textwrap.fill(line, width=70, subsequent_indent="    ")
            print(f"  {wrapped}")
    print(f"\n  {c('dim', f'⟨ generated in {elapsed}s ⟩')}")


def run_chapter(chapter: dict, quick: bool = False):
    section(
        f"Chapter {chapter['num']}: {chapter['title']}",
        icon=chapter["icon"],
    )
    print(f"  {c('dim', chapter['subtitle'])}\n")

    demos = chapter["demos"][:1] if quick else chapter["demos"]

    for demo in demos:
        demo_header(demo["label"], demo["lang"])
        print()
        for line in demo["prompt"].split("\n")[:5]:
            print(f"  {c('dim', line)}")
        if demo["prompt"].count("\n") > 5:
            print(f"  {c('dim', '...')}")
        print(f"\n  {c('yellow', '●')} {c('bold', 'TIMPS-Coder v2:')}\n")

        output, elapsed = run_model(demo["prompt"])
        show_output(output, elapsed)
        print()


def interactive_chat():
    section("Interactive Chat — TIMPS-Coder v2", icon="💬")
    print(f"  {c('dim', 'Ask anything: bug fix, algorithm, code review, planning.')}")
    print(f"  {c('dim', 'Type  exit  or  quit  to leave.')}\n")

    history = []

    while True:
        try:
            user_input = input(c("cyan", "  You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print(f"\n  {c('dim', 'Session ended.')}")
            break
        if not user_input:
            continue

        print(f"\n  {c('yellow', '●')} {c('bold', 'TIMPS-Coder v2:')}\n")
        output, elapsed = run_model(user_input)
        show_output(output, elapsed)
        print()


def print_capabilities():
    section("What Makes TIMPS-Coder v2 Different", icon="✦")
    caps = [
        ("Bug Analysis",       "Root-cause reasoning — not just pattern matching"),
        ("SWE Agent",          "Understands GitHub issues + multi-file context"),
        ("Algorithm Depth",    "Solves competitive problems with O-notation analysis"),
        ("Code Review",        "Catches security (SQL injection, XSS) + perf issues"),
        ("Agentic Planning",   "Plans multi-step workflows with exact commands"),
        ("THINK→ACT→VERIFY",   "Structured output: reason first, then code, then check"),
    ]
    for title, desc in caps:
        print(f"  {c('green', '✓')}  {c('bold', f'{title:<22}')}  {c('dim', desc)}")
    print()
    print(f"  {c('dim', 'Trained on: SWE-bench · SWE-Next · LeetCode · Agentic traces')}")
    print(f"  {c('dim', 'Base: Qwen2.5-Coder-0.5B · LoRA rank=16 · 3000 iters · Mac M2')}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TIMPS-Coder v2 Showcase")
    parser.add_argument("--chat",    action="store_true", help="Interactive chat mode")
    parser.add_argument("--quick",   action="store_true", help="One demo per chapter")
    parser.add_argument("--chapter", type=int, choices=range(1, 6),
                        metavar="N", help="Run single chapter (1–5)")
    args = parser.parse_args()

    # Check adapters exist
    if not Path(ADAPTER_PATH).exists():
        print(f"\n❌  Adapters not found: {ADAPTER_PATH}")
        print("    Train first:  bash 2_train_sft.sh\n")
        sys.exit(1)
    if not Path(ADAPTER_PATH, "adapters.safetensors").exists():
        print(f"\n❌  LoRA weights not found in: {ADAPTER_PATH}")
        print("    Train first:  bash 2_train_sft.sh\n")
        sys.exit(1)

    # Print banner
    print(c("purple", BANNER))
    print(f"  {c('bold', 'v2')}  Built by Sandeep Reddy  ·  TIMPS  ·  May 2026")
    print(f"  {c('dim', 'Base: Qwen2.5-Coder-0.5B · LoRA rank=16 · Agentic SFT')}\n")

    if args.chat:
        interactive_chat()
        return

    if args.chapter:
        chapter = next((ch for ch in CHAPTERS if ch["num"] == args.chapter), None)
        if chapter:
            run_chapter(chapter, quick=args.quick)
        return

    # Full showcase
    print_capabilities()

    chapters_to_run = CHAPTERS
    for chapter in chapters_to_run:
        run_chapter(chapter, quick=args.quick)

    section("Showcase Complete", icon="🎉")
    print(f"  {c('green', '✅')}  TIMPS-Coder v2 demo finished.\n")
    print(f"  Next steps:")
    print(f"  {c('dim', '  python3 3_benchmark_v2.py')}  — full 25-test benchmark vs base model")
    print(f"  {c('dim', '  python3 launch_timps_v2.py --chat')}  — interactive session")
    print(f"  {c('dim', '  python3 publish.py')}  — push to HuggingFace")
    print(f"  {c('dim', '  python3 4_make_gguf.py')}  — build Ollama GGUF\n")


if __name__ == "__main__":
    main()
