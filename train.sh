#!/bin/sh

# The default values
DEFAULT_DIM=25
DEFAULT_EPOCH=10
DEFAULT_LEARNING_RATE=1e-1
DEFAULT_BATCH_SIZE=30
DEFAULT_REG=1e-6

# Values to test
DIM_LIST=( 10 20 25 30 40 50 )
LEARNING_RATE_LIST=( 1e-4 1e-3 1e-2 1e-1 1 )
BATCH_SIZE_LIST=( 1 10 30 50 70 100 )
REG_LIST=( 1e-6 1e-4 1e-2 0 10 )

optimizer="adagrad"     # This is like constant

# Tune vector size
#------------------

epochs=$DEFAULT_EPOCH
learning_rate=$DEFAULT_LEARNING_RATE
batch_size=$DEFAULT_BATCH_SIZE
reg=$DEFAULT_REG
datetime=$(date +"%Y%m%d%H%M")
for dim in "${DIM_LIST[@]}"; do
    outfile="models/RNTN_D${dim}_E${epochs}_B${batch_size}_L${learning_rate}_R${reg}_${optimizer}_${datetime}.pickle"
    set -x
    python3 main.py \
        --dim=${dim} \
        --epochs=${epochs} \
        --learning-rate=${learning_rate} \
        --batch-size=${batch_size} \
        --reg=${reg} \
        --model=${outfile}
    set +x
done

# Tune batch size
#---------------------

epochs=$DEFAULT_EPOCH
learning_rate=$DEFAULT_LEARNING_RATE
dim=$DEFAULT_DIM
reg=$DEFAULT_REG
datetime=$(date +"%Y%m%d%H%M")
for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    outfile="models/RNTN_D${dim}_E${epochs}_B${batch_size}_L${learning_rate}_R${reg}_${optimizer}_${datetime}.pickle"
    set -x
    python3 main.py \
        --dim=${dim} \
        --epochs=${epochs} \
        --learning-rate=${learning_rate} \
        --batch-size=${batch_size} \
        --reg=${reg} \
        --model=${outfile}
    set +x
done

# Tune regularization parameter
#-------------------------------

epochs=$DEFAULT_EPOCH
learning_rate=$DEFAULT_LEARNING_RATE
dim=$DEFAULT_DIM
batch_size=$DEFAULT_BATCH_SIZE
datetime=$(date +"%Y%m%d%H%M")
for reg in "${REG_LIST[@]}"; do
    outfile="models/RNTN_D${dim}_E${epochs}_B${batch_size}_L${learning_rate}_R${reg}_${optimizer}_${datetime}.pickle"
    set -x
    python3 main.py \
        --dim=${dim} \
        --epochs=${epochs} \
        --learning-rate=${learning_rate} \
        --batch-size=${batch_size} \
        --reg=${reg} \
        --model=${outfile}
    set +x
done
