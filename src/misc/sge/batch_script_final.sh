#!/bin/bash

# qsub -l h_rt=48:00:00,mem_free=20,longq=1 -pe shm 32 -v SIZE="500",Q="0" batch_script_final.sh

PREFIX=${PREFIX:-"FINAL"}
SIZE=${SIZE:-"500"}
WINDOW=${WINDOW:-"10"}
NEG=${NEG:-"12"}
EPOCHS=${EPOCHS:-"50"}
SAMPLE=${SAMPLE:-"1e-4"}
DATASET=${DATASET:-"text8"}
Q=${Q:-"1"} # Quantization level default 1 bit
MIN_COUNT=${MIN_COUNT:-"5"} # Delete word if less than 5 occurrences (default)
OUTDIR=${OUTDIR:-"/home/maxlam/"}

/home/maxlam/word2bits_mikolov -save-every-epoch 0 -train /home/maxlam/${DATASET} -output ${OUTDIR}/${PREFIX}_vectors_dataset${DATASET}_epochs${EPOCHS}_size${SIZE}_neg${NEG}_window${WINDOW}_sample${SAMPLE}_Q${Q}_mincount${MIN_COUNT}.bin -size ${SIZE} -window ${WINDOW} -negative ${NEG} -sample ${SAMPLE} -threads 16 -binary 1 -iter ${EPOCHS} -min-count ${MIN_COUNT} -bitlevel ${Q} > ${OUTDIR}/${PREFIX}_vectors_1bit_dataset${DATASET}_epochs${EPOCHS}_size${SIZE}_neg${NEG}_window${WINDOW}_sample${SAMPLE}_Q${Q}_mincount${MIN_COUNT}_out

