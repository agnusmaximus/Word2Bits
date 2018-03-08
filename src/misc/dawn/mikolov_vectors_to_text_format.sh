#!/bin/bash

dir=/dfs/scratch0/maxlam/w2bfinalvectors/
files=`ls ${dir} | grep -v "txt"`

for filename in ${files}
do
    echo $filename
    python ~/Word2Bits/src/compress_word2bits.py ${dir}/${filename} ${dir}/${filename}_txt &
done
