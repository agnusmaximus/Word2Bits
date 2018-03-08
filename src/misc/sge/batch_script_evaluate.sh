#!/bin/bash

# qsub -l h_rt=0:30:00,mem_free=3,longq=0 -pe shm 1 -v OUTDIR=${outdir},INP=${file} batch_script_evaluate.sh

OUTDIR=${OUTDIR:-"./"}

/home/maxlam/compute_accuracy_mikolov $INP < /home/maxlam/questions-words.txt > ${INP}_evaluated_output

