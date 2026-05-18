# TIMPS-Coder v3 — Elite Bug-Fixing Assistant (0.5B)

> A 0.5B parameter coding model fine-tuned to **think before it codes** — specialising in bug
> analysis, code review, algorithm problem-solving, and agentic planning.  
> Built by [Sandeep Reddy](https://github.com/Sandeeprdy1729) · TIMPS · Made in India 🇮🇳

[![HuggingFace](https://img.shields.io/badge/HuggingFace-TIMPS--Coder--0.5B-yellow)](https://huggingface.co/sandeeprdy1729/TIMPS-Coder-0.5B)
[![Ollama](https://img.shields.io/badge/Ollama-sandeeprdy1729%2Ftimps--coder-blue)](https://ollama.com/sandeeprdy1729/timps-coder)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Internal%20Benchmark-88%25%20(44%2F50)-brightgreen)](benchmark_results.json)

---

## Model Summary

| Field | Value |
|---|---|
| **Base model** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Alibaba Cloud) |
| **Architecture** | Qwen2 Transformer — 494M parameters |
| **Fine-tuning method** | LoRA (rank=16, 16 layers) via MLX-LM |
| **Context window** | 4096 tokens |
| **Quantization** | Q4_K_M GGUF (Ollama) / BF16 safetensors (HuggingFace) |
| **Chat template** | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |
| **License** | Apache 2.0 |
| **Training hardware** | Apple M-series (Mac M1/M2/M3, 8 GB RAM) |

---

## Benchmark Results — 25 Tests, 5 Dimensions

> Evaluated on [`3_benchmark_ollama.py`](3_benchmark_ollama.py) — 25 hand-crafted tasks covering
> the most common real-world coding scenarios.  
> Scoring: **2 pts** = complete correct answer with code · **1 pt** = partial · **0** = wrong/refused.

| Dimension | Score | % | What is tested |
|---|---|---|---|
| 🐛 Bug Fix | 9 / 10 | **90%** | NullPointer, KeyError, off-by-one, async/await, recursion base case |
| 🔧 SWE / Repo-level | 9 / 10 | **90%** | ConcurrentModification, race conditions, N+1 queries, memory leaks, goroutine leaks |
| ⚡ Algorithms | 9 / 10 | **90%** | Two Sum O(n), sliding window, binary search rotated array, LRU Cache, merge K lists |
| 🔍 Code Review | 8 / 10 | **80%** | SQL injection, O(n²) → O(n), missing try/catch, mutable defaults, StringBuilder |
| 🤖 Agentic Reasoning | 9 / 10 | **90%** | Debug plan, GitHub Actions CI, monolith refactor, flaky test fix, profiling plan |
| **TOTAL** | **44 / 50** | **88%** | |

Full per-test results in [`benchmark_results.json`](benchmark_results.json).

### Reproduce the benchmark

```bash
# Requires Ollama running with the model pulled
ollama pull sandeeprdy1729/timps-coder
python3 3_benchmark_ollama.py          # full 25-test run (~2 min)
python3 3_benchmark_ollama.py --quick  # 10-test fast run
```

---

## Why 0.5B Can Beat Larger Models on This Task

A 0.5B model running locally beats cloud-called 7B models at bug fixing when:

1. **Narrow scope** — trained exclusively on bug fixing and code review, not general chat
2. **Format discipline** — every training sample uses `THINK → FIX → VERIFY` structure
3. **Low temperature** — `temp=0.1` minimises hallucination at inference time
4. **Fast feedback** — 2–6 second response, fully offline, no data leaves your machine
5. **Specialisation beats scale** at focused tasks

---

## Quick Start

### Option 1 — Ollama (recommended)

```bash
ollama pull sandeeprdy1729/timps-coder
ollama run sandeeprdy1729/timps-coder
```

### Option 2 — MLX (Mac Apple Silicon, no quantization loss)

```bash
pip install mlx-lm
mlx_lm.generate \
  --model sandeeprdy1729/TIMPS-Coder-0.5B \
  --max-tokens 700 --temp 0.1 \
  --prompt '<|im_start|>system
You are TIMPS-Coder v3. THINK through the root cause, FIX with complete code, VERIFY edge cases.<|im_end|>
<|im_start|>user
Fix the race condition: two threads increment self.count += 1 simultaneously.<|im_end|>
<|im_start|>assistant
'
```

### Option 3 — Python (HuggingFace Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")
tokenizer = AutoTokenizer.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")

messages = [
    {"role": "system",  "content": "You are TIMPS-Coder v3. THINK, FIX, VERIFY."},
    {"role": "user",    "content": "Fix: `data['user']['email']` throws KeyError when email is absent."},
]
text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=700, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Capabilities

| Does well | Limitations |
|---|---|
| Bug root-cause analysis with explanation | Complex multi-file refactors beyond ~300 lines |
| SQL injection, race condition, memory leak detection | May miss subtle business-logic bugs |
| O-notation analysis and algorithm optimisation | Not a replacement for static analysis tools |
| LeetCode medium-level algorithm problems | Struggles with competitive programming hard problems |
| GitHub Actions / CI YAML generation | Not trained on cloud IaC (Terraform, CDK) |
| Structured THINK → FIX → VERIFY responses | Always verify generated code before production use |

---

## Training Details

### Fine-tuning Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Fine-tuning method | LoRA (Supervised Fine-Tuning) |
| LoRA rank | 16 |
| LoRA target layers | 16 |
| Learning rate | 5e-6 |
| Iterations | 3,000 |
| Batch size | 1 (grad accumulation ×4 = effective batch 4) |
| Max sequence length | 2048 tokens |
| Framework | MLX-LM on Apple Silicon |
| Peak RAM | ~5.5 GB |
| Training time | ~3–4 hours on M2 Air 8 GB |

### Training Data

| Dataset | Type | Approx. Samples |
|---|---|---|
| `newfacade/LeetCodeDataset` | Algorithm problems with solutions | ~2,500 |
| `SWE-bench/SWE-bench_Verified` | Real GitHub issue → patch | ~400 |
| `TIGER-Lab/SWE-Next-SFT-Trajectories` | Agentic edit traces | ~2,000 |
| `WaltonFuture/agentic-sft-new` | Tool use + bash planning | ~3,000 |
| Custom TIMPS bug-fix corpus | Hand-curated bug/fix pairs | ~500 |
| **Total** | | **~8,400 samples** |

All samples formatted in ChatML. Answers structured as `THINK → FIX → VERIFY`.

### Data Format (ChatML)

```json
{
  "text": "<|im_start|>system\nYou are TIMPS-Coder v3...<|im_end|>\n<|im_start|>user\nFix: ...<|im_end|>\n<|im_start|>assistant\n**THINK:** ...\n\n**FIX:**\n```python\n...\n```\n\n**VERIFY:** ...<|im_end|>"
}
```

---

## Retrain It Yourself

### Requirements

- Mac M1 / M2 / M3, 8 GB+ RAM
- Python 3.10+

```bash
pip install mlx-lm datasets huggingface_hub
```

### Steps

```bash
git clone https://github.com/Sandeeprdy1729/TIMPS-Coder
cd timps-coder-finetune

# 1. Prepare training data
python3 1_prepare_data_v2.py

# 2. Fine-tune (~3-4 hours on M2 Air)
bash 2_train_sft_light.sh

# 3. Run benchmark
python3 3_benchmark_ollama.py

# 4. Convert to GGUF + push to Ollama
python3 4_make_gguf.py

# 5. Push to HuggingFace
huggingface-cli login
python3 publish.py
```

---

## Project Structure

```
timps-coder-finetune/
├── 1_prepare_data_v2.py     # Dataset builder
├── 2_train_sft_light.sh     # LoRA fine-tuning script
├── 2b_test_model.py         # Quick interactive model tester
├── 3_benchmark_ollama.py    # 25-test benchmark via Ollama API
├── 3_benchmark_v2.py        # 25-test benchmark via mlx_lm
├── 4_make_gguf.py           # Convert HF model to GGUF for Ollama
├── launch_timps_v2.py       # Live showcase + interactive chat REPL
├── prepare_new_data_v2.py   # CoT data generator (MBPP, HumanEval, commitpackft)
├── publish.py               # Push to HuggingFace + Ollama
├── Modelfile                # Ollama model config
├── benchmark_results.json   # Latest benchmark scores (25 tests)
├── data/
│   └── processed/
│       ├── train.jsonl      # 3,575 training samples (ChatML)
│       └── valid.jsonl
└── adapters/
    └── adapter_config.json  # LoRA adapter config
```

---

## Usage Tips

**Always use the system prompt** — the model performs best with it:

```
<|im_start|>system
You are TIMPS-Coder v3. THINK through the root cause, FIX with complete code, VERIFY edge cases.<|im_end|>
```

**Temperature**: Keep at `0.1` — higher values increase hallucination on a 0.5B model.

**Context**: Include the full function/class, not just the error message.

**Verification**: Always test generated code. Even at 88% accuracy, edge cases exist.

---

## About TIMPS

TIMPS-Coder is a personal/indie project exploring how far a tiny model can go on a focused
task through careful fine-tuning and structured training data.

- GitHub: [Sandeeprdy1729](https://github.com/Sandeeprdy1729)
- HuggingFace: [sandeeprdy1729](https://huggingface.co/sandeeprdy1729)
- YouTube: [@sandeepreddythummala](https://youtube.com/@sandeepreddythummala)

---

## License

Apache 2.0 — free to use, modify, and distribute commercially.  
Base model ([Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct))
is also Apache 2.0.
