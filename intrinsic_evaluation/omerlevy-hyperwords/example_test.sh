#!/bin/sh

# Download and install word2vecf
if [ ! -f word2vecf ]; then
    scripts/install_word2vecf.sh
fi


# Download corpus. We chose a small corpus for the example, and larger corpora will yield better results.
wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
gzip -d news.2010.en.shuffled.gz
CORPUS=news.2010.en.shuffled

# Clean the corpus from non alpha-numeric symbols
scripts/clean_corpus.sh $CORPUS > $CORPUS.clean


# Create two example collections of word-context pairs:

# A) Window size 2 with "clean" subsampling
mkdir w2.sub
python hyperwords/corpus2pairs.py --win 2 --sub 1e-5 ${CORPUS}.clean > w2.sub/pairs
scripts/pairs2counts.sh w2.sub/pairs > w2.sub/counts
python hyperwords/counts2vocab.py w2.sub/counts

# B) Window size 5 with dynamic contexts and "dirty" subsampling
mkdir w5.dyn.sub.del
python hyperwords/corpus2pairs.py --win 5 --dyn --sub 1e-5 --del ${CORPUS}.clean > w5.dyn.sub.del/pairs
scripts/pairs2counts.sh w5.dyn.sub.del/pairs > w5.dyn.sub.del/counts
python hyperwords/counts2vocab.py w5.dyn.sub.del/counts

# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 0.75 w2.sub/counts w2.sub/pmi
python hyperwords/counts2pmi.py --cds 0.75 w5.dyn.sub.del/counts w5.dyn.sub.del/pmi


# Create embeddings with SVD
python hyperwords/pmi2svd.py --dim 500 --neg 5 w2.sub/pmi w2.sub/svd
cp w2.sub/pmi.words.vocab w2.sub/svd.words.vocab
cp w2.sub/pmi.contexts.vocab w2.sub/svd.contexts.vocab
python hyperwords/pmi2svd.py --dim 500 --neg 5 w5.dyn.sub.del/pmi w5.dyn.sub.del/svd
cp w5.dyn.sub.del/pmi.words.vocab w5.dyn.sub.del/svd.words.vocab
cp w5.dyn.sub.del/pmi.contexts.vocab w5.dyn.sub.del/svd.contexts.vocab


# Create embeddings with SGNS (A). Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf -train w2.sub/pairs -pow 0.75 -cvocab w2.sub/counts.contexts.vocab -wvocab w2.sub/counts.words.vocab -dumpcv w2.sub/sgns.contexts -output w2.sub/sgns.words -threads 10 -negative 15 -size 500;
python hyperwords/text2numpy.py w2.sub/sgns.words
rm w2.sub/sgns.words
python hyperwords/text2numpy.py w2.sub/sgns.contexts
rm w2.sub/sgns.contexts

# Create embeddings with SGNS (B). Commands 2-5 are necessary for loading the vectors with embeddings.py
word2vecf/word2vecf -train w5.dyn.sub.del/pairs -pow 0.75 -cvocab w5.dyn.sub.del/counts.contexts.vocab -wvocab w5.dyn.sub.del/counts.words.vocab -dumpcv w5.dyn.sub.del/sgns.contexts -output w5.dyn.sub.del/sgns.words -threads 10 -negative 15 -size 500;
python hyperwords/text2numpy.py w5.dyn.sub.del/sgns.words
rm w5.dyn.sub.del/sgns.words
python hyperwords/text2numpy.py w5.dyn.sub.del/sgns.contexts
rm w5.dyn.sub.del/sgns.contexts


# Evaluate on Word Similarity
echo
echo "WS353 Results"
echo "-------------"

python hyperwords/ws_eval.py --neg 5 PPMI w2.sub/pmi testsets/ws/ws353.txt
python hyperwords/ws_eval.py --eig 0.5 SVD w2.sub/svd testsets/ws/ws353.txt
python hyperwords/ws_eval.py --w+c SGNS w2.sub/sgns testsets/ws/ws353.txt

python hyperwords/ws_eval.py --neg 5 PPMI w5.dyn.sub.del/pmi testsets/ws/ws353.txt
python hyperwords/ws_eval.py --eig 0.5 SVD w5.dyn.sub.del/svd testsets/ws/ws353.txt
python hyperwords/ws_eval.py --w+c SGNS w5.dyn.sub.del/sgns testsets/ws/ws353.txt


# Evaluate on Analogies
echo
echo "Google Analogy Results"
echo "----------------------"

python hyperwords/analogy_eval.py PPMI w2.sub/pmi testsets/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD w2.sub/svd testsets/analogy/google.txt
python hyperwords/analogy_eval.py SGNS w2.sub/sgns testsets/analogy/google.txt

python hyperwords/analogy_eval.py PPMI w5.dyn.sub.del/pmi testsets/analogy/google.txt
python hyperwords/analogy_eval.py --eig 0 SVD w5.dyn.sub.del/svd testsets/analogy/google.txt
python hyperwords/analogy_eval.py SGNS w5.dyn.sub.del/sgns testsets/analogy/google.txt
