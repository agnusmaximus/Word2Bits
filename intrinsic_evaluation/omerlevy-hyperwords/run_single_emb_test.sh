#!/bin/bash

# Usage: bash run_single_emb_test.sh PATH_TO_WORD_VECTORS OUTPUT_DIRECTORY

path=$1
outdir=$2
emb=`basename "${path}"`

ws_tasks=(ws353_similarity.txt ws353_relatedness.txt bruni_men.txt radinsky_mturk.txt luong_rare.txt simlex999.txt)
ws_taskfolder=testsets/ws

an_tasks=(google.txt msr.txt)

an_taskfolder=testsets/analogy

for task in ${ws_tasks[*]};
do
    echo "python hyperwords/ws_eval.py GLOVE ${path} ${ws_taskfolder}/${task} >> ${outdir}/${emb}_${task}_evalout"
    python hyperwords/ws_eval.py GLOVE ${path} ${ws_taskfolder}/${task} > ${outdir}/${emb}_${task}_evalout 
done
for task in ${an_tasks[*]};
do
    echo "python hyperwords/analogy_eval.py GLOVE ${path} ${an_taskfolder}/${task} >> ${outdir}/${emb}_${task}_evalout"
    python hyperwords/analogy_eval.py GLOVE ${path} ${an_taskfolder}/${task} > ${outdir}/${emb}_${task}_evalout 
done
