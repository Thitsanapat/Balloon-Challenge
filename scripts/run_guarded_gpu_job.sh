#!/usr/bin/env bash
# Run one GPU job while enforcing the shared-server thermal/power agreement.
set -euo pipefail

gpu_index="${BPC_GPU_INDEX:-0}"
max_temp_c="${BPC_MAX_TEMP_C:-75}"
other_util_limit="${BPC_OTHER_GPU_UTIL_LIMIT:-20}"
check_seconds="${BPC_GPU_CHECK_SECONDS:-5}"
log_path="${BPC_GPU_LOG:-gpu_guard.csv}"

if [[ "$gpu_index" != "0" ]]; then
    echo "Only Tesla GPU 0 is authorized for this project" >&2
    exit 2
fi
if [[ "$#" -eq 0 ]]; then
    echo "usage: run_guarded_gpu_job.sh COMMAND [ARG ...]" >&2
    exit 2
fi

# Prevent two jobs launched through this guard from occupying both Tesla cards.
exec 9>/tmp/balloon_challenge_one_tesla.lock
if ! flock -n 9; then
    echo "Another guarded Balloon Challenge GPU job is already running" >&2
    exit 4
fi

other_index=$((1 - gpu_index))
selected_name=$(nvidia-smi --id="$gpu_index" \
    --query-gpu=name --format=csv,noheader)
if [[ "$selected_name" != "Tesla P40" ]]; then
    echo "GPU $gpu_index is '$selected_name', not the permitted Tesla P40" >&2
    exit 5
fi
selected_memory=$(nvidia-smi --id="$gpu_index" \
    --query-gpu=memory.used --format=csv,noheader,nounits)
if (( selected_memory > 100 )); then
    echo "GPU $gpu_index is already in use (${selected_memory} MiB); refusing to start" >&2
    exit 3
fi

printf 'timestamp,gpu,temp_c,power_w,util_pct,memory_mib,other_gpu,other_util_pct\n' \
    > "$log_path"
# CUDA's default FASTEST_FIRST order can number a newer RTX card ahead of the
# Tesla cards even when nvidia-smi calls it GPU 2. PCI_BUS_ID keeps both tools'
# physical indices aligned, which is essential for monitoring the right card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu_uuid=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)
if [[ "$gpu_uuid" != GPU-* ]]; then
    echo "Could not identify Tesla 0 UUID; refusing to launch" >&2
    exit 5
fi
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
# Give the workload its own process group so any dataloader/worker children are
# stopped together if a safety limit is reached.
setsid -- "$@" &
child_pid=$!

stop_child() {
    if kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
}
trap stop_child INT TERM EXIT

while kill -0 "$child_pid" 2>/dev/null; do
    IFS=, read -r temperature power utilization memory_used < <(
        nvidia-smi --id="$gpu_index" \
            --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
            --format=csv,noheader,nounits
    )
    other_utilization=$(nvidia-smi --id="$other_index" \
        --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    temperature=${temperature// /}
    power=${power// /}
    utilization=${utilization// /}
    memory_used=${memory_used// /}
    other_utilization=${other_utilization// /}
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date --iso-8601=seconds)" "$gpu_index" "$temperature" "$power" \
        "$utilization" "$memory_used" "$other_index" "$other_utilization" \
        >> "$log_path"

    if (( temperature >= max_temp_c )); then
        echo "Stopping: GPU $gpu_index reached ${temperature}C" >&2
        stop_child
        exit 75
    fi
    if (( other_utilization >= other_util_limit )); then
        echo "Stopping: the other Tesla GPU reached ${other_utilization}% utilization" >&2
        stop_child
        exit 76
    fi
    sleep "$check_seconds"
done

set +e
wait "$child_pid"
status=$?
set -e
trap - INT TERM EXIT
exit "$status"
