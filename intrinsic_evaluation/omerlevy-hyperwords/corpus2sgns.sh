#!/bin/sh


# Download and install word2vecf
if [ ! -f word2vecf ]; then
    scripts/install_word2vecf.sh
fi


# Parse input params
PARAM_CHECK=$(python hyperwords/corpus2sgns_params.py $@ 2>&1)
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
WORD2VECF_OPTS=${PARAM_CHECK[3]}
SGNS2TEXT_OPTS=${PARAM_CHECK[4]}


# Clean the corpus from non alpha-numeric symbols
scripts/clean_corpus.sh $CORPUS > $CORPUS.clean


# Create collection of word-context pairs
mkdir $OUTPUT_DIR
python hyperwords/corpus2pairs.py $CORPUS2PAIRS_OPTS $CORPUS.clean > $OUTPUT_DIR/pairs
scripts/pairs2counts.sh $OUTPUT_DIR/pairs > $OUTPUT_DIR/counts
python hyperwords/counts2vocab.py $OUTPUT_DIR/counts


# Create embeddings with SGNS. Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf $WORD2VECF_OPTS -train $OUTPUT_DIR/pairs -cvocab $OUTPUT_DIR/counts.contexts.vocab -wvocab $OUTPUT_DIR/counts.words.vocab -dumpcv $OUTPUT_DIR/sgns.contexts -output $OUTPUT_DIR/sgns.words
python hyperwords/text2numpy.py $OUTPUT_DIR/sgns.words
python hyperwords/text2numpy.py $OUTPUT_DIR/sgns.contexts


# Save the embeddings in the textual format 
python hyperwords/sgns2text.py $SGNS2TEXT_OPTS $OUTPUT_DIR/sgns $OUTPUT_DIR/vectors.txt


# Remove temporary files
#rm $CORPUS.clean
#rm $OUTPUT_DIR/pairs
#rm $OUTPUT_DIR/counts*
#rm $OUTPUT_DIR/sgns*
