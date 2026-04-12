"""
TIMPS-Coder — Publish to HuggingFace + Ollama
Run: python3 publish.py  (after test.py passes)
"""
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

HF_USERNAME = "sandeeprdy1729"
MODEL_NAME  = "TIMPS-Coder-0.5B"
MODEL_PATH  = "./timps-coder-fused"
BASE_MODEL  = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
REPO_ID     = f"{HF_USERNAME}/{MODEL_NAME}"

bench = {}
if Path("test_results.json").exists():
    bench = json.load(open("test_results.json")).get("summary", {})

BASE_PCT  = bench.get("base_pct",  "—")
TUNED_PCT = bench.get("tuned_pct", "—")
WINS      = bench.get("wins",      "—")

CARD = f"""---
language: [en]
license: apache-2.0
base_model: {BASE_MODEL}
tags: [code, bug-fixing, qwen2.5-coder, mlx, timps, apple-silicon]
pipeline_tag: text-generation
---

# TIMPS-Coder-0.5B 🛠️
Fine-tuned coding model for bug fixing by [Sandeep Reddy](https://github.com/Sandeeprdy1729) · TIMPS Brand · Made in India 🇮🇳

## Benchmark
| Model | Score |
|-------|-------|
| Base ({BASE_MODEL}) | {BASE_PCT}% |
| **TIMPS-Coder (this)** | **{TUNED_PCT}%** |
*10-task bug-fix benchmark: NullPointer, KeyError, AsyncBug, ScopeBug, RecursionError, etc.*

## Quickstart — MLX (Mac M1/M2/M3)
```bash
pip install mlx-lm
mlx_lm generate --model {REPO_ID} --max-tokens 500 --temp 0.1 \\
  --prompt '<|im_start|>system
You are TIMPS-Coder. Explain the bug cause then show fixed code.<|im_end|>
<|im_start|>user
Fix null_pointer: My Spring @Autowired field is null<|im_end|>
<|im_start|>assistant'
```

## Training
- Base: `{BASE_MODEL}` | Method: LoRA rank=16 | HW: Mac M2 Air 8GB
- Dataset: 30,000+ clean coding instruction pairs (Python, Java, JS, C++, Go, Rust)
- Framework: MLX-LM | LR: 5e-6 | Iters: 3000

## Links
- GitHub: [Sandeeprdy1729](https://github.com/Sandeeprdy1729)  
- YouTube: [@sandeepreddythummala](https://youtube.com/@sandeepreddythummala)

Apache 2.0 License
"""

if not Path(MODEL_PATH).exists() or not (Path(MODEL_PATH)/"config.json").exists():
    print(f"❌ Model missing at {MODEL_PATH}. Run retrain.sh first.")
    sys.exit(1)

with open(f"{MODEL_PATH}/README.md","w") as f: f.write(CARD)
print("✅ Model card written")

api = HfApi()
create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
print(f"✅ Repo: https://huggingface.co/{REPO_ID}")
print("⏳ Uploading...")
upload_folder(folder_path=MODEL_PATH, repo_id=REPO_ID, repo_type="model",
    commit_message="🚀 TIMPS-Coder — clean trained release",
    ignore_patterns=["*.pyc","__pycache__"])
print(f"✅ Live: https://huggingface.co/{REPO_ID}")

mf = f'FROM {MODEL_PATH}\nSYSTEM """You are TIMPS-Coder by Sandeep Reddy. Fix bugs: explain root cause then show correct code."""\nPARAMETER temperature 0.1\nPARAMETER num_ctx 4096\n'
open("Modelfile","w").write(mf)
print("\nOllama:")
print("  ollama create sandeeprdy1729/timps-coder -f Modelfile")
print("  ollama push  sandeeprdy1729/timps-coder")