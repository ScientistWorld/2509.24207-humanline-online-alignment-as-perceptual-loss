"""One-off generator for eval/test/test_prompts.json (held-out test slice).

Pairs are uniform over three-digit operands and exclude every visible pair
(training / base-building pools and the visible validation slice), so test
inputs exist only in this directory.  The committed JSON is authoritative.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
sys.path.insert(0, str(WORKSPACE))
from data.arithmetic import visible_pairs  # noqa: E402

SEED = 863590777
SIZE = 4000

if __name__ == "__main__":
    exclude = visible_pairs() | {tuple(p) for p in json.loads((WORKSPACE / "eval/train/val_prompts.json").read_text())}
    rng = random.Random(SEED)
    out, seen = [], set()
    while len(out) < SIZE:
        pair = (rng.randint(0, 999), rng.randint(0, 999))
        if pair in exclude or pair in seen:
            continue
        seen.add(pair)
        out.append(list(pair))
    (HERE / "test_prompts.json").write_text(json.dumps(out) + "\n")
    print(f"wrote {len(out)} test pairs")
