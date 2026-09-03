#!/bin/bash
P=/home/sam/miniconda3/envs/heritage-opt/bin/python
cd /home/sam/paper_ws/paper-npj/拓本反演
$P -u src/exp_real.py      > results/_real.log     2>&1
$P -u src/exp_identify.py  > results/_ident.log    2>&1
$P -u src/exp_rate.py --seeds 3 > results/_rate.log 2>&1
$P -u src/exp_late.py      > results/_late.log     2>&1
echo ALLDONE
