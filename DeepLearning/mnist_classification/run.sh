#!/bin/bash
if [ -z $1 ]||[ $1 == "train" ];then
rm ./logs/*
echo "remove log files "
rm ./model/*
echo "remove saved models"
echo "start training!"
python3 train.py --epochs=10\
                --batch-size=64\
                --optimizer='momentum'\
                --validation-size=5000\
                --eval-frequency=100\
                --save\
                # --use-gpu               
elif [ $1 == "test" ];then
echo "start testing"
python3 test.py
else
echo "wrong command!"
fi
