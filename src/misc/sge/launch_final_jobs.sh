#!/bin/bash

epochs=( 1 10 25 50 )
#epochs=( 50 )
dimensions=( 100 200 400 600 800 1000 )
precisions=( 1 2 4 0 )

#epochs=( 50 )
#dimensions=( 1000 )
#precisions=( 4 )
OUTDIR=/home/maxlam/final_w2b_text8_neg24/
NEG=24

for epoch in "${epochs[@]}"
do
    for dimension in "${dimensions[@]}"
    do
	for precision in "${precisions[@]}"
	do
	    echo $epoch $dimension $precision
	    qsub -l h_rt=10:00:00,mem_free=5,longq=1 -pe shm 16 -v SIZE=${dimension},Q=${precision},EPOCHS=${epoch},OUTDIR=${OUTDIR},NEG=${NEG} batch_script_final.sh
	done
    done
done
