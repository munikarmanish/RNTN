#!/bin/sh

set -x

dataset="dev"
file=$1

python3 main.py --test \
    --dataset=${dataset} \
    --model=${file}
