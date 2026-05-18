#!/bin/bash
# Launch training in background — immune to Ctrl+C and terminal close
cd "$(dirname "$0")"
nohup bash 2_train_sft.sh >training.log 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > training.pid
echo "Training started  PID=$TRAIN_PID"
echo "Monitor:  tail -f training.log"
echo "Stop:     kill \$(cat training.pid)"
