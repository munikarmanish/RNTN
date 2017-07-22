#!/bin/sh

set -x

dim=25
epochs=30
learning_rate=1e-2
batch_size=30
optimizer="adagrad"
reg=1e-6
datetime=$(date +"%Y%m%d%H%M")
dataset="train"

outfile="models/RNTN_D${dim}_E${epochs}_B${batch_size}_L${learning_rate}_R${reg}_${optimizer}_${datetime}.pickle"

python3 main.py \
    --dim=${dim} \
    --epochs=${epochs} \
    --learning-rate=${learning_rate} \
    --batch-size=${batch_size} \
    --dataset=${dataset} \
    --model=${outfile}
