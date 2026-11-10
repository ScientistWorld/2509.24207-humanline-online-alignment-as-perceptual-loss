#!/bin/bash
# Helper sourced by method.sh / baseline.sh / ablation.sh: run training jobs in
# the background, at most $PARALLEL at a time, each logging to its run dir.
#   launch <run_dir> <python module> <args...>
#   finish                      # wait for the remaining jobs; fails if any failed

PARALLEL="${PARALLEL:-4}"
_pids=()

_run_one() {
  local dir="$1"; shift
  if ! python -m "$@" --run-dir "${dir}" > "${dir}/stdout.log" 2>&1; then
    echo "training FAILED: ${dir} (log: ${dir}/stdout.log)" >&2
    tail -n 20 "${dir}/stdout.log" >&2
    return 1
  fi
}

launch() {
  local dir="$1"; shift
  rm -rf "${dir}"
  mkdir -p "${dir}"
  _run_one "${dir}" "$@" &
  _pids+=("$!")
  if [ "${#_pids[@]}" -ge "${PARALLEL}" ]; then
    wait "${_pids[0]}"
    _pids=("${_pids[@]:1}")
  fi
}

finish() {
  local p
  for p in "${_pids[@]}"; do
    wait "${p}"
  done
  _pids=()
}
