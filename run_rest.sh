#!/bin/bash
P=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd /home/sam/paper_ws/paper-npj/拓本反演
$P -u src/exp_dating.py > results/_dating.log 2>&1
echo DATINGDONE
