# TIMPS-Coder v2 — Agentic Software Engineer

> A 0.5B coding model that **reasons before it codes** — trained on real GitHub issue resolutions,
> expert agent execution traces, competitive algorithms, and multi-step tool-use trajectories.  
> Built by [Sandeep Reddy](https://github.com/Sandeeprdy1729) · TIMPS · Made in India 🇮🇳

[![HuggingFace](https://img.shields.io/badge/HuggingFace-TIMPS--Coder--0.5B-yellow)](https://huggingface.co/sandeeprdy1729/TIMPS-Coder-0.5B)
[![Ollama](https://img.shields.io/badge/Ollama-sandeeprdy1729%2Ftimps--coder-blue)](https://ollama.com/sandeeprdy1729/timps-coder)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## What makes v2 different from every other 0.5B code model

| Capability | TIMPS-Coder v2 | Generic code models |
|---|---|---|
| Bug analysis | Root-cause reasoning (explains WHY) | Pattern-match and replace |
| SWE / Repo-level | Understands GitHub issues + multi-file context | Single-function scope |
| Algorithms | Solves competitive problems with O-notation | Basic code completion |
| Code review | Catches security (SQL injection) + perf + bugs | Syntax only |
| Agentic planning | Plans multi-step tasks with exact commands | Single response |
| Output structure | **THINK → ACT → VERIFY** | Unstructured prose |

---

## Benchmark — 25 tests across 5 dimensions

| Dimension | Tests | Description |
|---|---|---|
| 🐛 Bug Fix | 5 | NullPointer, KeyError, Async, Closure, RecursionError |
| 🔧 SWE | 5 | N+1 queries, race conditions, memory leaks, goroutine leaks |
| ⚡ Algorithm | 5 | Two Sum, Sliding Window, Binary Search, LRU Cache, Merge K Lists |
| 🔍 Code Review | 5 | SQL injection, O(n²) optimisation, missing error handling |
| 🤖 Agentic | 5 | CI debugging, refactoring plans, flaky tests, profiling |

```
python3 3_benchmark_v2.py        # full 25-test benchmark vs base model
python3 3_benchmark_v2.py --quick  # 10-test fast version
```

---

## Run in 30 seconds

### Ollama (recommended)
```bash
ollama run sandeeprdy1729/timps-coder
```

### MLX — Mac M1/M2/M3
```bash
pip install mlx-lm
mlx_lm generate \
  --model sandeeprdy1729/TIMPS-Coder-0.5B \
  --max-tokens 700 --temp 0.1 \
  --prompt '<|im_start|>system
You are TIMPS-Coder v2. THINK through root cause, ACT with complete code, VERIFY edge cases.<|im_end|>
<|im_start|>user
Repository: myapp/backend
Issue: N+1 queries — loading 100 products hits the DB 101 times. Fix it.<|im_end|>
<|im_start|>assistant
'
```

### Python
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")
tokenizer = AutoTokenizer.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")

messages = [
    {"role": "system",  "content": "You are TIMPS-Coder v2. THINK, ACT, VERIFY."},
    {"role": "user",    "content": "Fix the race condition: two threads increment self.count += 1 simultaneously."},
]
text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=700, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Showcase

```bash
git clone https://github.com/Sandeeprdy1729/TIMPS-Coder
cd timps-coder-finetune
pip install mlx-lm datasets huggingface_hub

python3 launch_timps_v2.py            # full 5-chapter showcase
python3 launch_timps_v2.py --chat     # interactive REPL
python3 launch_timps_v2.py --chapter 3  # single chapter (1–5)
python3 launch_timps_v2.py --quick    # fast 1-demo-per-chapter version
```

---

## Retrain it yourself

### Requirements
- Mac M1 / M2 / M3 (8 GB+ RAM)
- Python 3.10+
- `pip install mlx-lm datasets huggingface_hub`

### Steps

```bash
# 1. Build v2 dataset (SWE-bench + agentic traces + LeetCode)
python3 1_prepare_data_v2.py

# 2. Train  (~3–4 hrs on M2 Air 8 GB)
bash 2_train_sft.sh

# 3. Benchmark
python3 3_benchmark_v2.py

# 4. Showcase
python3 launch_timps_v2.py

# 5. Publish
huggingface-cli login
python3 publish.py

# 6. (Optional) Build Ollama GGUF
python3 4_make_gguf.py
```

### Training config (v2)

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Method | LoRA |
| LoRA rank | **16** (was 8 in v1) |
| LoRA layers | 16 |
| Learning rate | 5e-6 |
| Iterations | 3,000 |
| Batch size | 1 + grad accum ×4 |
| **Max seq length** | **2048** (was 512 in v1) |
| Framework | MLX-LM (Apple Silicon) |
| Peak memory | ~5.5 GB |

### v2 Training datasets

| Dataset | Type | Samples |
|---------|------|---------|
| `SWE-bench/SWE-bench_Verified` | Real GitHub issue → patch | ~400 |
| `TIGER-Lab/SWE-Next-SFT-Trajectories` | Agentic edit traces | ~2,000 |
| `newfacade/LeetCodeDataset` | Algorithm problems | ~2,500 |
| `WaltonFuture/agentic-sft-new` | Tool use + bash | ~3,000 |
| TIMPS v1 processed data | Bug-fix identity | existing |

---

## Project structure

```
timps-coder-finetune/
├── 1_prepare_data_v2.py   # v2 dataset: SWE-bench + agentic + LeetCode
├── 1_prepare_data.py      # v1 dataset builder (legacy)
├── 2_train_sft.sh         # LoRA fine-tuning (rank=16, seq=2048, 3000 iters)
├── 3_benchmark_v2.py      # 25-test benchmark — 5 capability dimensions
├── 4_make_gguf.py         # Convert to GGUF for Ollama
├── launch_timps_v2.py     # Live showcase + interactive chat
├── publish.py             # Push to HuggingFace + Ollama
├── test.py                # v1 benchmark (10 tests, legacy)
├── gen_dataset.py         # v1 dataset builder (legacy)
├── data/
│   └── processed/
│       ├── train.jsonl    # generated by 1_prepare_data_v2.py
│       └── valid.jsonl
└── adapters/              # LoRA adapter weights (generated)
```

---

## About TIMPS

TIMPS-Coder v2 is part of the **TIMPS** ecosystem.

- 🔧 [TIMPS CLI](https://github.com/Sandeeprdy1729/timps) — coding agent with 3-layer memory
- 🤗 [HuggingFace](https://huggingface.co/sandeeprdy1729)
- ▶️ [YouTube](https://youtube.com/@sandeepreddythummala)

## License

Apache 2.0 — free to use, modify, and distribute commercially.