#!/bin/bash

emb=$1
folder=$2

ws_tasks=(ws353_similarity.txt ws353_relatedness.txt bruni_men.txt radinsky_mturk.txt luong_rare.txt simlex999.txt)
ws_taskfolder=testsets/ws

an_tasks=(google.txt msr.txt)

an_taskfolder=testsets/analogy

for task in ${ws_tasks[*]};
do
    python hyperwords/ws_eval.py GLOVE ${folder}/${emb} ${ws_taskfolder}/${task} >> ${emb}_${task}_evalout 
done
for task in ${an_tasks[*]};
do
    python hyperwords/analogy_eval.py GLOVE ${folder}/${emb} ${an_taskfolder}/${task} >> ${emb}_${task}_evalout 
done
