#!/usr/bin/env python3
"""
update_hf_card.py — Push README.md as HuggingFace model card.
Run:  python3 update_hf_card.py
Requires: huggingface_hub, HF_TOKEN set or huggingface-cli login done.
"""
from huggingface_hub import HfApi, ModelCard

HF_REPO = "sandeeprdy1729/TIMPS-Coder-0.5B"

# --- Model card YAML header (metadata) + content -------------------------
CARD_CONTENT = """\
---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
  - code
  - bug-fixing
  - code-review
  - qwen2
  - lora
  - mlx
  - ollama
  - chatml
pipeline_tag: text-generation
library_name: transformers
---

# TIMPS-Coder v3 — Elite Bug-Fixing Assistant (0.5B)

> A 0.5B parameter coding model fine-tuned to **think before it codes** — specialising in bug
> analysis, code review, algorithm problem-solving, and agentic planning.  
> Built by [Sandeep Reddy](https://github.com/Sandeeprdy1729) · TIMPS · Made in India 🇮🇳

[![HuggingFace](https://img.shields.io/badge/HuggingFace-TIMPS--Coder--0.5B-yellow)](https://huggingface.co/sandeeprdy1729/TIMPS-Coder-0.5B)
[![Ollama](https://img.shields.io/badge/Ollama-sandeeprdy1729%2Ftimps--coder-blue)](https://ollama.com/sandeeprdy1729/timps-coder)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Internal%20Benchmark-88%25%20(44%2F50)-brightgreen)](https://github.com/Sandeeprdy1729/TIMPS-Coder/blob/main/benchmark_results.json)

## Model Summary

| Field | Value |
|---|---|
| **Base model** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Alibaba Cloud) |
| **Architecture** | Qwen2 Transformer — 494M parameters |
| **Fine-tuning method** | LoRA (rank=16, 16 layers) via MLX-LM |
| **Context window** | 4096 tokens |
| **Quantization** | Q4_K_M GGUF (Ollama) / BF16 safetensors (HuggingFace) |
| **Chat template** | ChatML (`<|im_start|>` / `<|im_end|>`) |
| **License** | Apache 2.0 |
| **Training hardware** | Apple M-series (Mac M1/M2/M3, 8 GB RAM) |

## Benchmark Results — 25 Tests, 5 Dimensions

Evaluated on [3_benchmark_ollama.py](https://github.com/Sandeeprdy1729/TIMPS-Coder/blob/main/3_benchmark_ollama.py).  
Scoring: **2 pts** = complete correct answer with code · **1 pt** = partial · **0** = wrong/refused.

| Dimension | Score | % |
|---|---|---|
| 🐛 Bug Fix | 9 / 10 | **90%** |
| 🔧 SWE / Repo-level | 9 / 10 | **90%** |
| ⚡ Algorithms | 9 / 10 | **90%** |
| 🔍 Code Review | 8 / 10 | **80%** |
| 🤖 Agentic Reasoning | 9 / 10 | **90%** |
| **TOTAL** | **44 / 50** | **88%** |

## Quick Start

### Ollama (recommended)

```bash
ollama pull sandeeprdy1729/timps-coder
ollama run sandeeprdy1729/timps-coder
```

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")
tokenizer = AutoTokenizer.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")

messages = [
    {"role": "system",  "content": "You are TIMPS-Coder v3. THINK through the root cause, FIX with complete code, VERIFY edge cases."},
    {"role": "user",    "content": "Fix: `data['user']['email']` throws KeyError when email is absent."},
]
text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=700, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### MLX (Mac Apple Silicon)

```bash
pip install mlx-lm
mlx_lm.generate \\
  --model sandeeprdy1729/TIMPS-Coder-0.5B \\
  --max-tokens 700 --temp 0.1 \\
  --prompt '<|im_start|>system
You are TIMPS-Coder v3. THINK through the root cause, FIX with complete code, VERIFY edge cases.<|im_end|>
<|im_start|>user
Fix the race condition: two threads increment self.count += 1 simultaneously.<|im_end|>
<|im_start|>assistant
'
```

## Training Details

### Fine-tuning Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Fine-tuning method | LoRA (Supervised Fine-Tuning) |
| LoRA rank | 16 |
| Learning rate | 5e-6 |
| Iterations | 3,000 |
| Batch size | 1 (grad accum ×4) |
| Max sequence length | 2048 tokens |
| Framework | MLX-LM on Apple Silicon |
| Peak RAM | ~5.5 GB |

### Training Data

| Dataset | Type | Approx. Samples |
|---|---|---|
| `newfacade/LeetCodeDataset` | Algorithm problems with solutions | ~2,500 |
| `SWE-bench/SWE-bench_Verified` | Real GitHub issue → patch | ~400 |
| `TIGER-Lab/SWE-Next-SFT-Trajectories` | Agentic edit traces | ~2,000 |
| `WaltonFuture/agentic-sft-new` | Tool use + bash planning | ~3,000 |
| Custom TIMPS bug-fix corpus | Hand-curated bug/fix pairs | ~500 |
| **Total** | | **~8,400 samples** |

All samples formatted in ChatML with `THINK → FIX → VERIFY` answer structure.

## Capabilities

| Does well | Limitations |
|---|---|
| Bug root-cause analysis with explanation | Complex multi-file refactors |
| SQL injection, race condition, memory leak detection | May miss subtle business-logic bugs |
| O-notation analysis and algorithm optimisation | Not a replacement for static analysis tools |
| LeetCode medium-level algorithm problems | Hard competitive programming problems |
| GitHub Actions / CI YAML generation | Not trained on Terraform, CDK |

## Usage Tips

- **Temperature**: Keep at `0.1` — higher values increase hallucination on a 0.5B model
- **Context**: Include the full function/class when asking for a bug fix
- **Verification**: Always test generated code. Even at 88% accuracy, edge cases exist
- **System prompt**: Required for best results — see the Quick Start examples above

## Training Code

Full training pipeline available at:  
[https://github.com/Sandeeprdy1729/TIMPS-Coder](https://github.com/Sandeeprdy1729/TIMPS-Coder)

## License

Apache 2.0 — free to use, modify, and distribute commercially.  
Base model (Qwen2.5-Coder-0.5B-Instruct) is also Apache 2.0.
"""

def main():
    api = HfApi()
    
    print(f"Uploading model card to {HF_REPO}...")
    card = ModelCard(CARD_CONTENT)
    card.push_to_hub(HF_REPO)
    print(f"✅ Model card updated: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
