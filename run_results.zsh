#!/usr/bin/env zsh

rg 'Optimized Circuit:' -A4 > optmized_stats.txt
rg 'Depth' optmized_stats.txt | cut -d ' ' -f4 | awk '{sum += $1} END {print sum/NR}'
rg 'fidelity' optmized_stats.txt | cut -d ' ' -f5 | awk '{sum += $1} END {print sum/NR}'
