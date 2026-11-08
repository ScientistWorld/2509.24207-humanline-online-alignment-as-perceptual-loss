#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for case_name in \
  reference_seed7 reference_seed8 reference_seed9 \
  better_more_rl better_warmstart weaker_less_rl weaker_warmstart weaker_no_humanline \
  label_copy_shortcut metadata_zero_shortcut constant_negative_cost_shortcut \
  constant_task_shortcut cherry_pick_shortcut reward_artifact_shortcut \
  forged_scores_shortcut empty_output_shortcut copied_input_shortcut \
  wrong_labels_shortcut null_cost_shortcut; do
  # Save each result before moving on; unexpected failures do not discard others.
  if [ -f "redteam/${case_name}/result.json" ]; then
    echo "Saved result exists: ${case_name}"
    continue
  fi
  echo "Running ${case_name}"
  bash "redteam/${case_name}/run.sh" || echo "Case needs inspection: ${case_name}"
done
python -m redteam.summarize
