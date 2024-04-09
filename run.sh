#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input.wav> <output.wav> <time_stretch_ratio>"
  exit 1
fi
python3 sol.py "$@"