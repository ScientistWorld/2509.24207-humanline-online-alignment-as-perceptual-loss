from __future__ import annotations

from method.humanline_grpo import parse_args, run


if __name__ == "__main__":
    args = parse_args()
    args.humanline = False
    run(args)
