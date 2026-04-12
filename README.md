# TIMPS-Coder 🛠️

> Fine-tuned coding model for bug fixing — trained on a MacBook M2 Air (8GB RAM)  
> Built by [Sandeep Reddy](https://github.com/Sandeeprdy1729) · TIMPS Brand · Made in India 🇮🇳

[![HuggingFace](https://img.shields.io/badge/HuggingFace-TIMPS--Coder--0.5B-yellow)](https://huggingface.co/sandeeprdy1729/TIMPS-Coder-0.5B)
[![Ollama](https://img.shields.io/badge/Ollama-sandeeprdy1729%2Ftimps--coder-blue)](https://ollama.com/sandeeprdy1729/timps-coder)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## What it does

TIMPS-Coder fixes bugs across Java, Python, JavaScript, C++, Go, Rust and more.  
For every bug, it:
1. **Explains the root cause** in plain English
2. **Shows the complete corrected code**

## Benchmark

| Model | Score | Hardware |
|-------|-------|----------|
| Base (Qwen2.5-Coder-0.5B) | 88% | — |
| **TIMPS-Coder-0.5B** | **92%** | Mac M2 Air 8GB |

*10-task bug-fix benchmark: NullPointer, KeyError, IndexOutOfBounds, AsyncBug, ScopeBug, RecursionError, TypeError, ConcurrentModification, LogicError*  
Result: **3 wins · 6 ties · 1 loss**

## Run it in 30 seconds

### Ollama (recommended)
```bash
ollama run sandeeprdy1729/timps-coder
```

### MLX — Mac M1/M2/M3
```bash
pip install mlx-lm
mlx_lm generate \
  --model sandeeprdy1729/TIMPS-Coder-0.5B \
  --max-tokens 500 --temp 0.1 \
  --prompt '<|im_start|>system
You are TIMPS-Coder. Explain the bug cause then show fixed code.<|im_end|>
<|im_start|>user
Fix null_pointer: My Spring @Autowired field is null<|im_end|>
<|im_start|>assistant
'
```

### Python
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")
tokenizer = AutoTokenizer.from_pretrained("sandeeprdy1729/TIMPS-Coder-0.5B")

messages = [
    {"role": "system",  "content": "You are TIMPS-Coder. Explain the bug cause then show fixed code."},
    {"role": "user",    "content": "Fix null_pointer: My Spring @Autowired field is null"},
]
text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=500, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training — reproduce it yourself

Everything is open source. You can retrain this on your own Mac.

### Requirements
- Mac M1/M2/M3 (8GB+ RAM)
- Python 3.10+
- `pip install mlx-lm datasets huggingface_hub`

### Steps

```bash
# 1. Clone
git clone https://github.com/Sandeeprdy1729/TIMPS-Coder
cd TIMPS-Coder

# 2. Build clean dataset
python3 build_clean_dataset.py
# Downloads 30,000+ clean coding instruction pairs from HuggingFace

# 3. Fix code fences
python3 fix_fences.py

# 4. Train (~2 hrs on M2 Air)
bash retrain.sh

# 5. Test
python3 test.py

# 6. Publish
huggingface-cli login
python3 publish.py
```

### Training config
| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Method | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA layers | 8 |
| Learning rate | 5e-6 |
| Iterations | 3,000 |
| Batch size | 2 + grad accum ×4 |
| Max seq length | 1024 |
| Framework | MLX-LM (Apple Silicon) |
| Peak memory | 3.7 GB |

## Project structure

```
TIMPS-Coder/
├── build_clean_dataset.py   # Download + clean training data
├── fix_fences.py            # Fix missing code fences in dataset
├── retrain.sh               # LoRA fine-tuning script
├── test.py                  # 10-task benchmark vs base model
├── publish.py               # Push to HuggingFace + Ollama
├── data/
│   └── processed/
│       ├── train.jsonl      # Training data (generated)
│       └── valid.jsonl      # Validation data (generated)
└── adapters/                # LoRA adapter weights (generated)
```

## About TIMPS

TIMPS-Coder is part of the **TIMPS** ecosystem — a personal AI OS for developers built by Sandeep Reddy from Hyderabad, India.

- 🔧 [TIMPS CLI](https://github.com/Sandeeprdy1729) — coding agent with 3-layer memory
- 📺 [YouTube](https://youtube.com/@sandeepreddythummala) — n8n automation + AI tools
- 🤗 [HuggingFace](https://huggingface.co/sandeeprdy1729)

## License

Apache 2.0 — free to use, modify, and distribute commercially.