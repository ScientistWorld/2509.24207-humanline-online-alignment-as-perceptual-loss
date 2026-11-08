"""Baseline GRPO (online, 64x-stale, or offline) without humanline changes.

Same code path as the method with ``--humanline`` forced off: the reference
model is the policy that generated the current sampling round.
"""

from __future__ import annotations

from method.humanline_grpo import parse_args, run

if __name__ == "__main__":
    args = parse_args()
    args.humanline = False
    run(args)
