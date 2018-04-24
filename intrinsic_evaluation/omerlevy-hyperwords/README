# hyperwords: Hyperparameter-Enabled Word Representations #

hyperwords is a collection of scripts and programs for creating word representations, designed to facilitate academic
research and prototyping of word representations. It allows you to tune many hyperparameters that are pre-set or
ignored in other word representation packages.

hyperwords is free and open software. If you use hyperwords in scientific publication, we would appreciate citations:  
"Improving Distributional Similarity with Lessons Learned from Word Embeddings"
Omer Levy, Yoav Goldberg, and Ido Dagan. TACL 2015.


## Requirements ##
Running hyperwords may require a lot of computational resources:  
- disk space for independently pre-processing the corpus  
- internal memory for loading sparse matrices  
- significant running time; hyperwords is neither optimized nor multi-threaded

hyperwords assumes a *nix shell, and requires Python 2.7 (or later, excluding 3+) with the following packages installed:
numpy, scipy, sparsesvd, docopt.


## Quick-Start ##
1. Download the latest version from BitBucket, unzip, and make sure all scripts have running permissions (chmod 755 *.sh).
2. Download a text corpus of your choice.
3. To create word vectors...
    * ...with SVD over PPMI, use: *corpus2svd.sh*
    * ...with SGNS (skip-grams with negative sampling), use: *corpus2sgns.sh*
4. The vectors should be available in textual format under <output_path>/vectors.txt

To explore the list of hyperparameters, use the *-h* or *--help* option.


##Pipeline##
The following figure shows the hyperwords' pipeline:

**DATA:**  raw corpus  =>  corpus  =>  pairs  =>  counts  =>  vocab  
**TRADITIONAL:**  counts + vocab  =>  pmi  =>  svd  
**EMBEDDINGS:**  pairs  + vocab  =>  sgns  

**raw corpus  =>  corpus**  
- *scripts/clean_corpus.sh*
- Eliminates non-alphanumeric tokens from the original corpus.

**corpus  =>  pairs**  
- *corpus2pairs.py*  
- Extracts a collection of word-context pairs from the corpus.

**pairs  =>  counts**  
- *scripts/pairs2counts.sh*
- Aggregates identical word-context pairs.

**counts  =>  vocab**  
- *counts2vocab.py*  
- Creates vocabularies with the words' and contexts' unigram distributions.

**counts + vocab  =>  pmi**  
- *counts2pmi.py*  
- Creates a PMI matrix (*scipy.sparse.csr_matrix*) from the counts.

**pmi  =>  svd**  
- *pmi2svd.py*  
- Factorizes the PMI matrix using SVD. Saves the result as three dense numpy matrices.

**pairs  + vocab  =>  sgns**  
- *word2vecf/word2vecf*
- An external program for creating embeddings with SGNS. For more information, see:  
**"Dependency-Based Word Embeddings". Omer Levy and Yoav Goldberg. ACL 2014.**

An example pipeline is demonstrated in: *example_test.sh*


##Evaluation##
hyperwords also allows easy evaluation of word representations on two tasks: word similarity and analogies.

**Word Similarity**
- *hyperwords/ws_eval.py*
- Compares how a representation ranks pairs of related words by similarity versus human ranking.  
- 5 readily-available datasets

**Analogies**  
- *hyperwords/analogy_eval.py*
- Solves analogy questions, such as: "man is to woman as king is to...?" (answer: queen).  
- 2 readily-available datasets  
- Shows results of two analogy recovery methods: 3CosAdd and 3CosMul. For more information, see:  
**"Linguistic Regularities in Sparse and Explicit Word Representations". Omer Levy and Yoav Goldberg. CoNLL 2014.**

These programs assume that the representation was created by hyperwords, and can be loaded by
*hyperwords.representations.embedding.Embedding*. Dense vectors in textual format (such as the ones produced by word2vec
and GloVe) can be converted to hyperwords' format using *hyperwords/text2numpy.py*.