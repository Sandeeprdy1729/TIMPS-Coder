"""
TIMPS-Coder — Step 2b: Quick Test (run after training)
=======================================================
Run this to verify your model works before publishing.

Usage: python 2b_test_model.py
"""

import subprocess
import sys

MODEL_PATH = "./timps-coder-fused"

TEST_CASES = [
    {
        "label": "NullPointerException Fix",
        "prompt": "Fix null_pointer: I'm getting a NullPointerException when calling list.stream().map(x -> x.getName()).collect(Collectors.toList()). The list was returned from a database query.",
    },
    {
        "label": "IndexOutOfBounds Fix",
        "prompt": "Fix index_error: ArrayIndexOutOfBoundsException at position 5 in a loop iterating over an array of size 5 in Java.",
    },
    {
        "label": "Python KeyError Fix",
        "prompt": "Fix key_error: Python dict throws KeyError when accessing user['email'] even though I checked 'email' in user first.",
    },
]

SYSTEM = (
    "You are TIMPS-Coder, an expert coding assistant specialized in "
    "debugging and fixing code. Analyze the bug carefully, reason through "
    "the root cause, and provide a clean, correct fix with explanation."
)

def build_prompt(instruction: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def test_model():
    print("\n🧪 TIMPS-Coder Model Test")
    print("=" * 50)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}] {case['label']}")
        print("-" * 40)
        print(f"Input: {case['prompt'][:80]}...")

        prompt = build_prompt(case["prompt"])

        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.generate",
                "--model", MODEL_PATH,
                "--max-tokens", "512",
                "--prompt", prompt,
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout.strip()
        # Strip the prompt echo from output
        if "<|im_start|>assistant" in output:
            output = output.split("<|im_start|>assistant")[-1].strip()

        print(f"Output:\n{output[:600]}")
        print()

    print("✅ Tests complete! If outputs look sensible → run python 3_publish.py")

if __name__ == "__main__":
    test_model()
