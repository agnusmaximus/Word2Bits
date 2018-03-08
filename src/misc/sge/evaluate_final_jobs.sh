#!/bin/bash

outdir="final_w2b_text8_neg24/"

for file in ${outdir}/*bin
do
    echo $file
    qsub -l h_rt=0:30:00,mem_free=5,longq=0 -pe shm 1 -v OUTDIR=${outdir},INP=${file} batch_script_evaluate.sh
done
