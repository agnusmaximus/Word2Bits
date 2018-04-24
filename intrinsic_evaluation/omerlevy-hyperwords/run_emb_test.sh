#!/bin/bash

#Put your embedding folder here
#folder=/dfs/scratch1/senwu/embedding_evaluation/test/test/text8
#folder=/dfs/scratch0/maxlam/w2bfinalvectors/
folder=/dfs/scratch0/maxlam/w2bthresholdedfinalvectors/

#Put your embeddintg files that in the embedding folder here
embs=`ls ${folder} | grep txt`

ws_tasks=(ws353_similarity.txt ws353_relatedness.txt bruni_men.txt radinsky_mturk.txt luong_rare.txt simlex999.txt)
ws_taskfolder=testsets/ws

an_tasks=(google.txt msr.txt)
an_taskfolder=testsets/analogy

for emb in ${embs[*]};
do
    echo $emb
    bash run_single_emb_test.sh ${emb} ${folder} &
done

