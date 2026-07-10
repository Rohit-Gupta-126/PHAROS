#!/usr/bin/env bash
# Compile the standalone SOFIE trigger-score binary. Needs the generated
# header first (see README.md) plus g++ and OpenBLAS:
#   sudo apt-get install g++ libopenblas-dev
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f generated/encoder_mu.hxx ]]; then
    echo "generated/encoder_mu.hxx missing -- run scripts/sofie_probe.py in a" >&2
    echo "SOFIE-enabled ROOT first (docker/root-sofie.Dockerfile)." >&2
    exit 1
fi

g++ -O2 -std=c++17 -o sofie_score main.cpp -lopenblas
echo "built $(pwd)/sofie_score"
