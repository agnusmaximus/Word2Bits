#!/bin/sh


# Parse input params
PARAM_CHECK=$(python hyperwords/corpus2svd_params.py $@ 2>&1)
if [[ $PARAM_CHECK == *Usage:* ]]; then
    echo "$PARAM_CHECK";
    exit 1
fi

PARAM_CHECK=$(echo $PARAM_CHECK | tr '@ ' ' @')
PARAM_CHECK=($PARAM_CHECK)
for((i=0; i < ${#PARAM_CHECK[@]}; i++))
do
    PARAM_CHECK[i]=$(echo ${PARAM_CHECK[i]} | tr '@ ' ' @')
done

set -x #echo on
CORPUS=${PARAM_CHECK[0]}
OUTPUT_DIR=${PARAM_CHECK[1]}
CORPUS2PAIRS_OPTS=${PARAM_CHECK[2]}
COUNTS2PMI_OPTS=${PARAM_CHECK[3]}
PMI2SVD_OPTS=${PARAM_CHECK[4]}
SVD2TEXT_OPTS=${PARAM_CHECK[5]}


# Clean the corpus from non alpha-numeric symbols
scripts/clean_corpus.sh $CORPUS > $CORPUS.clean


# Create collection of word-context pairs
mkdir $OUTPUT_DIR
python hyperwords/corpus2pairs.py $CORPUS2PAIRS_OPTS $CORPUS.clean > $OUTPUT_DIR/pairs
scripts/pairs2counts.sh $OUTPUT_DIR/pairs > $OUTPUT_DIR/counts
python hyperwords/counts2vocab.py $OUTPUT_DIR/counts


# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py $COUNTS2PMI_OPTS $OUTPUT_DIR/counts $OUTPUT_DIR/pmi


# Create embeddings with SVD
python hyperwords/pmi2svd.py $PMI2SVD_OPTS $OUTPUT_DIR/pmi $OUTPUT_DIR/svd
cp $OUTPUT_DIR/pmi.words.vocab $OUTPUT_DIR/svd.words.vocab
cp $OUTPUT_DIR/pmi.contexts.vocab $OUTPUT_DIR/svd.contexts.vocab


# Save the embeddings in the textual format 
python hyperwords/svd2text.py $SVD2TEXT_OPTS $OUTPUT_DIR/svd $OUTPUT_DIR/vectors.txt


# Remove temporary files
#rm $CORPUS.clean
#rm $OUTPUT_DIR/pairs
#rm $OUTPUT_DIR/counts*
#rm $OUTPUT_DIR/pmi*
#rm $OUTPUT_DIR/svd*
