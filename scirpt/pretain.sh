#!/bin/bash
#SBATCH -J patchmix                   # 指定作业名
#SBATCH -o patchmix-pretrain-[lr=1*5e-4,batchsize=512].out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p 3090-shen                    # 作业提交的分区为 test
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=16      # 每个节点运行 32 个任务, 每个任务默认分配1个cpu
#SBATCH --gres=gpu:4          # 单个节点使用 n 块 GPU 卡

# 运行程序 并行任务
source ~/.bashrc  # 必须有，没这个跑不了
source activate vmae  #运行特定anconda环境
cd /home/scccse/lijt/project/contrastive_learning/patchmix
python main_pretrain.py