#!/bin/bash

# Rebuild
make clean
make

# Usage: ./compare.sh <INPUT>
INPUT="$1"

if [[ -z "$INPUT" ]]; then
    echo "Usage: $0 <INPUT file base name (without extension)>"
    exit 1
fi

# Run ./bench-a100 and save output
./bench-a100 "${INPUT}.fastq" "${INPUT}.fasta" > bench_a100_output.txt

# Run ./matcher and save output
./matcher "${INPUT}.fastq" "${INPUT}.fasta" > matcher_output.txt

if diff -q bench_a100_output.txt matcher_output.txt > /dev/null; then
    echo "same"
    exit 0
else
    echo "not the same"
    exit 1
fi
