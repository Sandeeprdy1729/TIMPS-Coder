"""
TIMPS-Coder — Step 4 (Optional): Convert to GGUF for Ollama 🦙
===============================================================
After publishing to HuggingFace, convert to GGUF so anyone
can run: ollama pull timps-coder

This uses llama.cpp's conversion tools.

Usage: python 4_make_gguf.py
"""

import os
import subprocess
import sys
from pathlib import Path

HF_REPO     = "sandeeprdy1729/TIMPS-Coder-0.5B"   # source HuggingFace model
MODEL_PATH  = "./timps-coder-hf"               # local download dir
OUTPUT_DIR  = "./timps-coder-gguf"
OLLAMA_NAME = "sandeeprdy1729/timps-coder"      # Ollama registry target
QUANT_TYPE  = "q4_k_m"   # Best quality/size balance — runs on 4GB RAM


def download_from_hf():
    """Download the latest model from HuggingFace to a local directory."""
    if Path(MODEL_PATH).exists() and any(Path(MODEL_PATH).glob("*.safetensors")):
        print(f"✅  Model already downloaded at {MODEL_PATH}")
        return
    print(f"⬇️  Downloading {HF_REPO} from HuggingFace...")
    # Use Python API directly — avoids deprecated CLI and interactive prompts
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"✅  Downloaded to {MODEL_PATH}")


def install_llamacpp():
    print("📦 Installing llama-cpp-python...")
    subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)


def convert_to_gguf():
    print("\n🔄 Converting to GGUF...")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Clone llama.cpp if not present
    if not Path("llama.cpp").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "--depth=1"],
            check=True
        )

    # Install only the conversion dependencies (avoid downgrading existing packages)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "gguf>=0.1.0", "sentencepiece"],
        check=True
    )

    # Derive a clean basename from the Ollama model name (e.g. "timps-coder")
    _base = OLLAMA_NAME.split("/")[-1]

    # Convert to float16 GGUF first
    f16_path = f"{OUTPUT_DIR}/{_base}-f16.gguf"
    if Path(f16_path).exists():
        print(f"  Step 1: F16 GGUF already exists, skipping → {f16_path}")
    else:
        print(f"  Step 1: Converting to F16 GGUF → {f16_path}")
        subprocess.run(
            [
                sys.executable, "llama.cpp/convert_hf_to_gguf.py",
                MODEL_PATH,
                "--outfile", f16_path,
                "--outtype", "f16",
            ],
            check=True,
        )

    # Quantize to Q4_K_M
    q4_path = f"{OUTPUT_DIR}/{_base}-{QUANT_TYPE}.gguf"
    print(f"\n  Step 2: Quantizing to {QUANT_TYPE.upper()} → {q4_path}")

    # Build quantize binary using CMake (Makefile removed from recent llama.cpp)
    build_dir = "llama.cpp/build"
    quantize_bin = f"{build_dir}/bin/llama-quantize"
    if not Path(quantize_bin).exists():
        print("  Building llama-quantize with CMake...")
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        # Ensure Homebrew cmake is on PATH
        env = {**os.environ, "PATH": "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")}
        subprocess.run(
            ["cmake", "llama.cpp", f"-B{build_dir}", "-DCMAKE_BUILD_TYPE=Release"],
            check=True, env=env
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--config", "Release",
             "--target", "llama-quantize", "-j4"],
            check=True, env=env
        )

    subprocess.run(
        [quantize_bin, f16_path, q4_path, QUANT_TYPE.upper()],
        check=True
    )

    print(f"\n✅ GGUF created: {q4_path}")
    return q4_path


def create_modelfile(gguf_path: str):
    """Create Ollama Modelfile."""
    modelfile = f"""FROM {gguf_path}

SYSTEM \"\"\"You are TIMPS-Coder v3, an elite coding assistant built by Sandeep Reddy (TIMPS).

For every coding task:
1. THINK: Analyse the problem and identify the root cause
2. ACT: Write clean, correct, well-documented code with proper error handling
3. VERIFY: Check edge cases, time/space complexity, security implications

Principles: Helpful · Harmless · Honest\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
"""
    modelfile_path = f"{OUTPUT_DIR}/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    print(f"\n📄 Modelfile created: {modelfile_path}")
    return modelfile_path


def push_to_ollama(modelfile_path: str):
    """Create the Ollama model locally then push to registry."""
    print(f"\n🦙 Creating Ollama model: {OLLAMA_NAME}")
    subprocess.run(["ollama", "create", OLLAMA_NAME, "-f", modelfile_path], check=True)
    print(f"✅  Model created locally")

    print(f"\n📤 Pushing to Ollama registry: https://ollama.com/{OLLAMA_NAME}")
    subprocess.run(["ollama", "push", OLLAMA_NAME], check=True)
    print(f"✅  Pushed: https://ollama.com/{OLLAMA_NAME}")


def main():
    print("\n🦙 TIMPS-Coder → GGUF → Ollama Pipeline")
    print("=" * 45)
    print(f"  Source : HuggingFace  {HF_REPO}")
    print(f"  Target : Ollama       {OLLAMA_NAME}")
    print("=" * 45)

    download_from_hf()
    gguf_path = convert_to_gguf()
    modelfile_path = create_modelfile(gguf_path)
    push_to_ollama(modelfile_path)

    print("\n✅  All done!")
    print(f"   Run locally : ollama run {OLLAMA_NAME}")
    print(f"   Public page : https://ollama.com/{OLLAMA_NAME}")


if __name__ == "__main__":
    main()
