#!/bin/bash

dir="final_w2b_text8_neg24"
files=`ls ${dir} | grep evaluated_output`

for f in $files
do
    # FINAL_vectors_1bit_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q2_mincount5_out  
    prefix=`echo $f | grep -oP "datasettext[0-9]+_epochs[0-9]+_size[0-9]+_neg[0-9]+_window[0-9]+_sample1e-4_Q[0-9]+_mincount5"`
    evaluated_outfile=$f
    loss_outfile="FINAL_vectors_1bit_${prefix}_out"
    last_loss=`cat ${dir}/$loss_outfile | tail -n 1 | grep -oP "Loss: (.+)"`
    google_acc=`cat ${dir}/$evaluated_outfile | grep "Questions seen" -B 1 | grep -oP "Total accuracy: .+ %"`
    #echo "${evaluated_outfile} ${last_loss}"
    echo ${evaluated_outfile} ${google_acc}
done

echo ""
echo ""
echo ""

for f in $files
do
    # FINAL_vectors_1bit_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q2_mincount5_out  
    prefix=`echo $f | grep -oP "datasettext[0-9]+_epochs[0-9]+_size[0-9]+_neg[0-9]+_window[0-9]+_sample1e-4_Q[0-9]+_mincount5"`
    evaluated_outfile=$f
    loss_outfile="FINAL_vectors_1bit_${prefix}_out"
    last_loss=`cat ${dir}/$loss_outfile | tail -n 1 | grep -oP "Loss: (.+)"`
    google_acc=`cat ${dir}/$evaluated_outfile | grep "Questions seen" -B 1 | grep -oP "Total accuracy: .+ %"`
    echo "${evaluated_outfile} ${last_loss}"
    #echo ${evaluated_outfile} ${google_acc}
done
	 
