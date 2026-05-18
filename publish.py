"""
TIMPS-Coder v2 — Publish to HuggingFace + Ollama
Run: python3 publish.py  (after 3_benchmark_v2.py passes)
"""
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

HF_USERNAME = "sandeeprdy1729"
MODEL_NAME  = "TIMPS-Coder-0.5B"
MODEL_PATH  = "./timps-coder-fused"
BASE_MODEL  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
REPO_ID     = f"{HF_USERNAME}/{MODEL_NAME}"

# Load v2 benchmark results if available, fall back to v1
bench = {}
for results_file in ["test_results_v2.json", "test_results.json"]:
    if Path(results_file).exists():
        bench = json.load(open(results_file)).get("summary", {})
        break

BASE_PCT   = bench.get("base_pct",   "—")
TUNED_PCT  = bench.get("tuned_pct",  "—")
DELTA_PCT  = bench.get("delta_pct",  "—")
WINS       = bench.get("wins",       "—")
TIES       = bench.get("ties",       "—")
LOSSES     = bench.get("losses",     "—")

CARD = f"""---
language: [en]
license: apache-2.0
base_model: {BASE_MODEL}
tags: [code, bug-fixing, code-agent, swe-bench, qwen2.5-coder, mlx, timps, apple-silicon, agentic]
pipeline_tag: text-generation
---

# TIMPS-Coder v2 — Agentic Software Engineer
Fine-tuned by [Sandeep Reddy](https://github.com/Sandeeprdy1729) · TIMPS · Made in India 🇮🇳

> **Not just a bug-fixer.** TIMPS-Coder v2 is a 0.5B agentic coding model trained on real
> GitHub issue resolutions (SWE-bench), expert agent execution traces, competitive algorithms,
> and multi-step tool-use trajectories. It reasons before it codes.

## What makes v2 different

| Capability | Description |
|---|---|
| 🐛 **Deep Bug Analysis** | Root-cause reasoning — explains WHY before showing the fix |
| 🔧 **SWE / Repo-Level** | Understands GitHub issues across multiple files |
| ⚡ **Algorithm Mastery** | Solves competitive problems with complexity analysis |
| 🔍 **Code Review Agent** | Finds security, performance & correctness issues together |
| 🤖 **Agentic Planning** | Plans multi-step tasks with exact commands, like a senior engineer |

## Benchmark (25 tests · 5 dimensions)

| Model | Bug Fix | SWE | Algo | Review | Agent | **Total** |
|-------|---------|-----|------|--------|-------|-----------|
| Base ({BASE_MODEL}) | — | — | — | — | — | **{BASE_PCT}%** |
| **TIMPS-Coder v2** | — | — | — | — | — | **{TUNED_PCT}%** |

*Result: {WINS} wins · {TIES} ties · {LOSSES} losses  (+{DELTA_PCT}% over base)*

## Training data (v2)

| Dataset | Type | Size |
|---------|------|------|
| SWE-bench/SWE-bench_Verified | Real GitHub issue → patch | ~400 |
| TIGER-Lab/SWE-Next-SFT-Trajectories | Agentic edit traces (ShareGPT) | ~2K |
| newfacade/LeetCodeDataset | Algorithm problems | ~2.5K |
| WaltonFuture/agentic-sft-new | Tool use + bash traces | ~3K |
| TIMPS v1 data | Bug-fix identity baseline | existing |

## Quickstart — MLX (Mac M1/M2/M3)
```bash
pip install mlx-lm
mlx_lm generate --model {REPO_ID} --max-tokens 700 --temp 0.1 \\
  --prompt '<|im_start|>system
You are TIMPS-Coder v2. THINK through root cause, ACT with complete code, VERIFY edge cases.<|im_end|>
<|im_start|>user
Repository: myapp/backend
Issue: N+1 queries — loading 100 products hits the DB 101 times.<|im_end|>
<|im_start|>assistant
'
```

## Quickstart — Python (transformers)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("{REPO_ID}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}")

messages = [
    {{"role": "system",  "content": "You are TIMPS-Coder v2. THINK, ACT, VERIFY."}},
    {{"role": "user",    "content": "Fix: my React useEffect has a memory leak from an event listener."}},
]
text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=700, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Ollama
```bash
ollama run sandeeprdy1729/timps-coder
```

## Training details
- Base: `{BASE_MODEL}` · LoRA rank=16 · 16 layers
- Iters: 3000 · LR: 5e-6 · seq-len: 2048 · batch: 1 (grad-accum 4)
- Hardware: Mac M2 Air 8GB · Framework: MLX-LM

## Showcase
```bash
git clone https://github.com/Sandeeprdy1729/TIMPS-Coder
cd timps-coder-finetune
python3 launch_timps_v2.py          # full demo
python3 launch_timps_v2.py --chat   # interactive
python3 3_benchmark_v2.py           # reproduce benchmark
```

## Links
- GitHub: [Sandeeprdy1729](https://github.com/Sandeeprdy1729)
- YouTube: [@sandeepreddythummala](https://youtube.com/@sandeepreddythummala)

Apache 2.0 License
"""

if not Path(MODEL_PATH).exists() or not (Path(MODEL_PATH) / "config.json").exists():
    print(f"❌  Model not found at {MODEL_PATH}")
    print("    Run first:  bash 2_train_sft.sh")
    sys.exit(1)

# Write model card
with open(f"{MODEL_PATH}/README.md", "w") as f:
    f.write(CARD)
print("✅  Model card written")

# Push to HuggingFace
api = HfApi()
create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
print(f"✅  Repo: https://huggingface.co/{REPO_ID}")
print("⏳  Uploading...")
upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="TIMPS-Coder v2 — agentic SFT release",
    ignore_patterns=["*.pyc", "__pycache__"],
)
print(f"✅  Live: https://huggingface.co/{REPO_ID}")

# Write Modelfile for Ollama
SYSTEM_OLLAMA = (
    "You are TIMPS-Coder v2, an agentic software engineer by Sandeep Reddy (TIMPS). "
    "For every task: THINK through root cause or approach, "
    "ACT with complete production-ready code, "
    "VERIFY edge cases and complexity."
)
mf = (
    f'FROM {MODEL_PATH}\n\n'
    f'SYSTEM """{SYSTEM_OLLAMA}"""\n\n'
    f'PARAMETER temperature 0.1\n'
    f'PARAMETER num_ctx 4096\n'
    f'PARAMETER stop "<|im_end|>"\n'
)
with open("Modelfile", "w") as f:
    f.write(mf)
print("\n  Ollama (run after GGUF is ready):")
print("    ollama create sandeeprdy1729/timps-coder -f Modelfile")
print("    ollama push  sandeeprdy1729/timps-coder")