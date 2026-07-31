#!/bin/bash
# Heterogeneous multi-core case study.
#
# Runs the six Figure 8 Pareto-optimal EENN architectures on the nine 4-core systems in
# stream/inputs/eenn/hardware/casestudy/. Every system has the same compute and memory budget
# and shares the paper's pooling, SIMD and off-chip cores, the same 2x2 mesh and the same
# mapping; only the mix of compute-core styles varies. sys_EEEE is the Figure 8 accelerator
# itself and anchors the numbers.
#
# The GA budget is 64 generations x 64 individuals, matching main_stream_eenn_nas.py, so the
# anchor is directly comparable to the published Figure 8 values.
#
# Parallel runs are independent: every output path is namespaced by hardware and workload, and
# nothing in the GA path writes to a shared location. Note the GA is not seeded, so re-running
# a completed job can give a slightly different allocation. That is the search, not the
# parallelism, and it is why finished runs are skipped rather than recomputed.
#
# Usage:
#   ./eenn_casestudy.sh [-j N] [--force] [--dry-run]
set -u -o pipefail

PYTHON=${PYTHON:-.venv/bin/python}
HW_DIR=stream/inputs/eenn/hardware/casestudy
NAS_DIR=stream/inputs/eenn/workload/nas
MAPPING=stream/inputs/eenn/mapping/edge_tpu_like_quad_core.yaml
OUT_ROOT=outputs-eenn/hw_casestudy
PB=8
PC=8
GENERATIONS=64
INDIVIDUALS=64

# The six Pareto-optimal architectures on the gold line of Figure 8.
# "<Figure 8 label>:<nas directory>:<complete exit list, final exit included>"
WORKLOADS=(
    "6_11_iter_0:iter_0/net_3:6,11"
    "2_8_11_iter_0:iter_0/net_8:2,8,11"
    "0_4_5_7_11_iter_0:iter_0/net_1:0,4,5,7,11"
    "2_3_4_5_9_11_iter_3:iter_3/net_3:2,3,4,5,9,11"
    "0_1_5_8_9_11_iter_2:iter_2/net_7:0,1,5,8,9,11"
    "2_3_4_5_6_9_11_iter_6:iter_6/net_0:2,3,4,5,6,9,11"
)

JOBS=$(( $(nproc) / 2 )); [ "$JOBS" -lt 1 ] && JOBS=1
FORCE=0
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --force)   FORCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -j)        JOBS="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
    shift
done

[ -x "$PYTHON" ] || { echo "error: interpreter '$PYTHON' not found" >&2; exit 2; }
cd "$(dirname "$0")" || exit 1

mapfile -t HW_FILES < <(find "$HW_DIR" -maxdepth 1 -name 'sys_*.yaml' | sort)
[ ${#HW_FILES[@]} -gt 0 ] || { echo "error: no systems in $HW_DIR" >&2; exit 2; }

run_one() {
    local hw_file=$1 label=$2 relpath=$3 ids=$4
    local hw_name model_str out_dir stage_pickle log_file
    hw_name=$(basename "$hw_file" .yaml)
    model_str=${ids//,/_}
    out_dir="$OUT_ROOT/$hw_name/pb${PB}_pc${PC}"
    stage_pickle="$out_dir/stage_data/model_${model_str}.pickle"
    log_file="$out_dir/logs/${label}.log"

    if [ -s "$stage_pickle" ] && [ "$FORCE" -eq 0 ]; then
        echo "SKIP  $hw_name  $label"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "WOULD $hw_name  $label"
        return 0
    fi

    mkdir -p "$out_dir/logs"
    echo "START $hw_name  $label"
    if "$PYTHON" main_stream_eenn.py \
            -id "$ids" -w "$NAS_DIR/$relpath/model.onnx" \
            -hw "$hw_file" -map "$MAPPING" -o "$OUT_ROOT" \
            -pb "$PB" -pc "$PC" -g "$GENERATIONS" -i "$INDIVIDUALS" &> "$log_file"; then
        if [ -s "$stage_pickle" ]; then
            echo "DONE  $hw_name  $label"
        else
            echo "FAIL  $hw_name  $label  (no stage data; see $log_file)"
            return 1
        fi
    else
        echo "FAIL  $hw_name  $label  (exit $?; see $log_file)"
        return 1
    fi
}

echo "systems=${#HW_FILES[@]}  workloads=${#WORKLOADS[@]}  ga=${GENERATIONS}x${INDIVIDUALS}  jobs=$JOBS"
echo "output root: $OUT_ROOT"
echo

for hw_file in "${HW_FILES[@]}"; do
    for entry in "${WORKLOADS[@]}"; do
        IFS=':' read -r label relpath ids <<< "$entry"
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
        run_one "$hw_file" "$label" "$relpath" "$ids" &
    done
done
wait

[ "$DRY_RUN" -eq 1 ] && exit 0

echo
echo "=== summary ==="
total=0
for hw_file in "${HW_FILES[@]}"; do
    hw_name=$(basename "$hw_file" .yaml)
    done_n=0
    for entry in "${WORKLOADS[@]}"; do
        IFS=':' read -r _ _ ids <<< "$entry"
        [ -s "$OUT_ROOT/$hw_name/pb${PB}_pc${PC}/stage_data/model_${ids//,/_}.pickle" ] && done_n=$((done_n + 1))
    done
    total=$((total + done_n))
    printf '  %-12s %d/%d\n' "$hw_name" "$done_n" "${#WORKLOADS[@]}"
done
echo "  total: $total/$(( ${#HW_FILES[@]} * ${#WORKLOADS[@]} ))"
[ "$total" -eq $(( ${#HW_FILES[@]} * ${#WORKLOADS[@]} )) ] || exit 1
