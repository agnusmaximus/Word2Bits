CC_MIKOLOV=g++
CC_GLOVE=gcc
CFLAGS=-Ofast -march=native -lm -lpthread -funroll-loops -Wno-unused-result

# Data file path
TEXT8_PATH := /afs/ir.stanford.edu/users/m/a/maxlam/text8
FULL_WIKI_PATH := /dfs/scratch0/maxlam/wiki.en.text

# Path to source code
MIKOLOV_DIR := ./src/mikolov/
GLOVE_DIR := ./src/glove/src/
GLOVE_EVAL_DIR := ./src/glove/eval/

# Path to binary outputs
BUILDDIR := ./build/

# Path to saved vector outputs
SAVEDIR := ./save_vectors/
FULLWIKISAVEDIR := /lfs/1/maxlam/w2b/

# Shared variables
SAVE_FILE=$(SAVEDIR)/vectors
WINDOW_SIZE=10
VECTOR_SIZE=400
NUM_THREADS=45
MAX_ITER=50

# Glove variables
QUESTION_DATA_PATH=./data/google_analogies_test_set/question-data/
VOCAB_MIN_COUNT=5
VERBOSE=2
VOCAB_FILE=$(BUILDDIR)/glove_vocab.txt
MEMORY=4.0
COOCCURRENCE_FILE=$(BUILDDIR)/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$(BUILDDIR)/cooccurrence.shuf.bin
BINARY=2
X_MAX=10

# Mikolov variables
MIKOLOV_SAVE_FILE=$(SAVEDIR)/mikolov_vectors.bin
FULLWIKI_MIKOLOV_SAVE_FILE=$(FULLWIKISAVEDIR)/mikolov_vectors.bin
NEGATIVE_SIZE=12
MIN_COUNT=5
MIN_COUNT_FULL=5

# QUANTIZATION VARIABLES
BITLEVEL=0

benchmark-glove:
	@echo Building Glove...
	$(CC_GLOVE) $(GLOVE_DIR)/glove.c -o $(BUILDDIR)/glove $(CFLAGS)
	$(CC_GLOVE) $(GLOVE_DIR)/shuffle.c -o $(BUILDDIR)/shuffle $(CFLAGS)
	$(CC_GLOVE) $(GLOVE_DIR)/cooccur.c -o $(BUILDDIR)/cooccur $(CFLAGS)
	$(CC_GLOVE) $(GLOVE_DIR)/vocab_count.c -o $(BUILDDIR)/vocab_count $(CFLAGS)

	@echo Running Glove...
	#$(BUILDDIR)/vocab_count -min-count $(VOCAB_MIN_COUNT) -verbose $(VERBOSE) < $(TEXT8_PATH) > $(VOCAB_FILE)
	#$(BUILDDIR)/cooccur -memory $(MEMORY) -vocab-file $(VOCAB_FILE) -verbose $(VERBOSE) -window-size $(WINDOW_SIZE) < $(TEXT8_PATH) > $(COOCCURRENCE_FILE)
	#$(BUILDDIR)/shuffle -memory $(MEMORY) -verbose $VERBOSE < $(COOCCURRENCE_FILE) > $(COOCCURRENCE_SHUF_FILE)
	$(BUILDDIR)/glove -save-file $(SAVE_FILE) -threads $(NUM_THREADS) -input-file $(COOCCURRENCE_SHUF_FILE) -x-max $X_MAX -iter $(MAX_ITER) -vector-size $(VECTOR_SIZE) -binary $(BINARY) -vocab-file $(VOCAB_FILE) -verbose $(VERBOSE)

	@echo Evaluating Glove...
	python $(GLOVE_EVAL_DIR)/evaluate.py --vocab_file $(VOCAB_FILE) --vectors_file $(SAVE_FILE).txt --question_data_path $(QUESTION_DATA_PATH)
benchmark-mikolov:
	@echo Building Mikolov...
	$(CC_MIKOLOV) -w  $(MIKOLOV_DIR)/compute-accuracy.c $(CFLAGS) -o $(BUILDDIR)/compute_accuracy_mikolov
	$(CC_MIKOLOV) $(MIKOLOV_DIR)/word2bits.cpp $(CFLAGS) -o $(BUILDDIR)/word2bits_mikolov

	@echo Running Mikolov...
	$(BUILDDIR)/word2bits_mikolov -min-count $(MIN_COUNT) -bitlevel ${BITLEVEL} -train $(TEXT8_PATH) -output $(MIKOLOV_SAVE_FILE)_text8_bitlevel${BITLEVEL}_size${VECTOR_SIZE}_window${WINDOW_SIZE}_neg${NEGATIVE_SIZE}_maxiter${MAX_ITER}_mincount${MIN_COUNT} -size $(VECTOR_SIZE) -window $(WINDOW_SIZE) -negative $(NEGATIVE_SIZE) -sample 1e-4 -threads $(NUM_THREADS) -binary 1 -iter $(MAX_ITER)

	@echo Evaluating Mikolov
	$(BUILDDIR)/compute_accuracy_mikolov $(MIKOLOV_SAVE_FILE)_text8_bitlevel${BITLEVEL}_size${VECTOR_SIZE}_window${WINDOW_SIZE}_neg${NEGATIVE_SIZE}_maxiter${MAX_ITER}_mincount${MIN_COUNT} < data/google_analogies_test_set/questions-words.txt
benchmark-mikolov-large:
	@echo Building Mikolov...
	$(CC_MIKOLOV) -w  $(MIKOLOV_DIR)/compute-accuracy.c $(CFLAGS) -o $(BUILDDIR)/compute_accuracy_mikolov
	$(CC_MIKOLOV) $(MIKOLOV_DIR)/word2bits.cpp $(CFLAGS) -o $(BUILDDIR)/word2bits_mikolov

	@echo Running Mikolov...
	$(BUILDDIR)/word2bits_mikolov -min-count $(MIN_COUNT_FULL) -bitlevel ${BITLEVEL} -train $(FULL_WIKI_PATH) -output $(FULLWIKI_MIKOLOV_SAVE_FILE)_fullwiki_bitlevel${BITLEVEL}_size${VECTOR_SIZE}_window${WINDOW_SIZE}_neg${NEGATIVE_SIZE}_maxiter${MAX_ITER}_mincount${MIN_COUNT_FULL} -size $(VECTOR_SIZE) -window $(WINDOW_SIZE) -negative $(NEGATIVE_SIZE) -sample 1e-4 -threads $(NUM_THREADS) -binary 1 -iter $(MAX_ITER)

	@echo Evaluating Mikolov
	$(BUILDDIR)/compute_accuracy_mikolov $(FULLWIKI_MIKOLOV_SAVE_FILE)_fullwiki_bitlevel${BITLEVEL}_size${VECTOR_SIZE}_window${WINDOW_SIZE}_neg${NEGATIVE_SIZE}_maxiter${MAX_ITER}_mincount${MIN_COUNT_FULL} < data/google_analogies_test_set/questions-words.txt
clean:
	rm -f $(BUILDDIR)/word2bits_mikolov
	rm -f $(BUILDDIR)/compute_accuracy_mikolov
	rm -f $(BUILDDIR)/glove
	rm -f $(BUILDDIR)/shuffle
	rm -f $(BUILDDIR)/cooccur
