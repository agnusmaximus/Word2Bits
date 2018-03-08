#!/bin/bash


SEED=1234
EPOCH=50
BATCH_SIZE=32

folder="SQuAD"
embtypes=`ls ${folder} | grep -oP "mikolov_vectors.bin_fullwiki_bitlevel[0-9]+_size[0-9]+_window[0-9]+_neg[0-9]+_maxiter[0-9]+_mincount[0-9]+_txt" | sort | uniq`
#embtypes=`ls ${folder} | grep -oP "mikolov_vectors.bin_fullwiki_bitlevel[0-9]+_size[0-9]+_window[0-9]+_neg[0-9]+_maxiter[0-9]+_mincount[0-9]+_txt" | sort | uniq | grep mikolov_vectors.bin_fullwiki_bitlevel2_size1000_window10_neg12_maxiter25_mincount5`

for embtype in ${embtypes[*]};
do    

    mkdir -p models/${embtype}_${SEED}
    mkdir -p train_log/${embtype}_${SEED}

    export PYTHONPATH=../../sru/

    python train.py --meta_file ${folder}/${embtype}-meta.msgpack --data_file ${folder}/${embtype}-data.msgpack --train_orig ${folder}/${embtype}-train.csv --dev_orig ${folder}/${embtype}-dev.csv --seed ${SEED} --epochs ${EPOCH} --batch_size ${BATCH_SIZE} --tune_partial 0 --fix_embeddings --model_dir /lfs/1/maxlam/drqamodels/${embtype}_${SEED} --log_file train_log/${embtype}_${SEED}/output.log
done

