#!/usr/bin/env zsh

OUTPUT_DIR="run_outputs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
NUM_RUNS=800
MAX_JOBS=42

echo "Running $NUM_RUNS iterations with $MAX_JOBS parallel jobs..."

for ((i=1; i<=NUM_RUNS; i++)); do
    printf -v run_num "%03d" $i
    (uv run visualize.py > "$OUTPUT_DIR/output_${run_num}.txt" 2>&1) &

    if (( (i % MAX_JOBS) == 0 )); then
        wait
    fi
    printf "\rProgress: [%-50s] %d%%" "$(printf '#%.0s' {1..$((i*50/NUM_RUNS))})" $((i*100/NUM_RUNS))
done
wait
echo "All runs completed!"
