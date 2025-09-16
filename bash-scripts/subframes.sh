#!/bin/bash

# Usage: ./subframes.sh <lammps trajectory file> <sample_rate> <output.dump>

filename="$1"
fourth_line=$(sed -n '4p' "$filename")
one_frame=$((fourth_line + 9))
sample_rate=$2
out_file="$3"
big_block=$((sample_rate * one_frame))

awk -v e="$one_frame" -v step="$big_block" 'NR % step >= 1 && NR % step <= e' $filename > $out_file

