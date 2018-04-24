#Put your embedding folder here
folder=/dfs/scratch1/senwu/embedding_evaluation/test/test/text8

#Put your embeddintg files that in the embedding folder here
embs=(vectors_d50.txt.e0.004315_i50.final)

ws_tasks1=(UMNSRS-rel.txt UMNSRS-sim.txt)
ws_taskfolder1=testsets/ws/UMNSRS

ws_task2=MayoSRS.txt
ws_taskfolder2=testsets/ws/MayoSRS

for emb in ${embs[*]};
do
    for task in ${ws_tasks1[*]};
    do
#        echo ${task} >> $1
        python hyperwords/ws_eval_ignore_nonexist.py GLOVE ${folder}/${emb} ${ws_taskfolder1}/${task} >> $1
    done
    python hyperwords/ws_eval_ignore_nonexist.py GLOVE ${folder}/${emb} ${ws_taskfolder2}/${ws_task2} >> $1
done

