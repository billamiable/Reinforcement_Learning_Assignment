#!/usr/bin/env bash

# python train_policy.py 'pm' --exp_name p2_60_r --history 60 --discount 0.90 -lr 5e-4 -n 60 -rec
python train_policy.py 'pm' --exp_name p2_50_r --history 50 --discount 0.90 -lr 5e-4 -n 60 -rec
python train_policy.py 'pm' --exp_name p2_40_r --history 40 --discount 0.90 -lr 5e-4 -n 60 -rec
python train_policy.py 'pm' --exp_name p2_30_r --history 30 --discount 0.90 -lr 5e-4 -n 60 -rec
python train_policy.py 'pm' --exp_name p2_60 --history 60 --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name p2_50 --history 50 --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name p2_40 --history 40 --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name p2_30 --history 30 --discount 0.90 -lr 5e-4 -n 60