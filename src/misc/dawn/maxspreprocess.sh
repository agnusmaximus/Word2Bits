#!/bin/bash

folder=/dfs/scratch0/maxlam/w2bfinalvectors/
embs=`ls ${folder} | grep txt`
#embs=`ls ${folder} | grep txt | grep mikolov_vectors.bin_fullwiki_bitlevel0_size1000_window10_neg12_maxiter25_mincount5`

EMBFOLDER=/lfs/1/maxlam/
LOGFOLDER=log

for emb in ${embs[*]};
do
    dimension=`echo ${emb} | grep -oP "size([0-9]+)" | grep -oP "([0-9]+)"`
    echo $emb
    echo $dimension
    echo ${folder}/${emb}
    python prepro.py --wv_type ${emb} --sample_size 1000 --wv_dim ${dimension} --wv_file ${folder}/${emb}
done

