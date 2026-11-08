"""One-off generator for eval/train/val_prompts.json (visible validation slice).

Pairs are uniform over three-digit operands and exclude every pair in the
visible training / base-building pools.  The committed JSON is authoritative.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from data.arithmetic import visible_pairs  # noqa: E402

SEED = 777001
SIZE = 2000

if __name__ == "__main__":
    exclude = visible_pairs()
    rng = random.Random(SEED)
    out, seen = [], set()
    while len(out) < SIZE:
        pair = (rng.randint(0, 999), rng.randint(0, 999))
        if pair in exclude or pair in seen:
            continue
        seen.add(pair)
        out.append(list(pair))
    (HERE / "val_prompts.json").write_text(json.dumps(out) + "\n")
    print(f"wrote {len(out)} validation pairs")
