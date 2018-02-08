CC_MIKOLOV=g++
CC_GLOVE=gcc
CFLAGS=-Ofast -march=native -lm -lpthread -funroll-loops -Wno-unused-result

# Data file path
TEXT8_PATH := /afs/ir.stanford.edu/users/m/a/maxlam//text8

# Path to source code
MIKOLOV_DIR := ./src/mikolov/
GLOVE_DIR := ./src/glove/src/
GLOVE_EVAL_DIR := ./src/glove/eval/

# Path to binary outputs
BUILDDIR := ./build/

# Path to saved vector outputs
SAVEDIR := ./save_vectors/

# Shared variables
SAVE_FILE=$(SAVEDIR)/vectors
WINDOW_SIZE=15
VECTOR_SIZE=200
NUM_THREADS=8
MAX_ITER=2

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

benchmark-glove:
	@echo Building Glove...
	$(CC_GLOVE) $(GLOVE_DIR)/glove.c -o $(BUILDDIR)/glove $(CFLAGS)
	$(CC_GLOVE) $(GLOVE_DIR)/shuffle.c -o $(BUILDDIR)/shuffle $(CFLAGS)
	$(CC_GLOVE) $(GLOVE_DIR)/cooccur.c -o $(BUILDDIR)/cooccur $(CFLAGS)
	$(CC_GLOVE) $(GLOVE_DIR)/vocab_count.c -o $(BUILDDIR)/vocab_count $(CFLAGS)

	@echo Running Glove...
	$(BUILDDIR)/vocab_count -min-count $(VOCAB_MIN_COUNT) -verbose $(VERBOSE) < $(TEXT8_PATH) > $(VOCAB_FILE)
	$(BUILDDIR)/cooccur -memory $(MEMORY) -vocab-file $(VOCAB_FILE) -verbose $(VERBOSE) -window-size $(WINDOW_SIZE) < $(TEXT8_PATH) > $(COOCCURRENCE_FILE)
	$(BUILDDIR)/shuffle -memory $(MEMORY) -verbose $VERBOSE < $(COOCCURRENCE_FILE) > $(COOCCURRENCE_SHUF_FILE)
	$(BUILDDIR)/glove -save-file $(SAVE_FILE) -threads $(NUM_THREADS) -input-file $(COOCCURRENCE_SHUF_FILE) -x-max $X_MAX -iter $(MAX_ITER) -vector-size $(VECTOR_SIZE) -binary $(BINARY) -vocab-file $(VOCAB_FILE) -verbose $(VERBOSE)

	@echo Evaluating Glove...
	python $(GLOVE_EVAL_DIR)/evaluate.py --vocab_file $(VOCAB_FILE) --vectors_file $(SAVE_FILE).txt --question_data_path $(QUESTION_DATA_PATH)
benchmark-mikolov:
	@echo Building Mikolov...
	$(CC_MIKOLOV) -w  $(MIKOLOV_DIR)/compute-accuracy.c $(CFLAGS) -o $(BUILDDIR)/compute_accuracy_mikolov
	$(CC_MIKOLOV) $(MIKOLOV_DIR)/word2bits.cpp $(CFLAGS) -o $(BUILDDIR)/word2bits_mikolov

	@echo Running Mikolov...
	$(BUILDDIR)/word2bits_mikolov -train $(TEXT8_PATH) -output $(SAVEDIR)/mikolov-vectors.bin -size $(VECTOR_SIZE) -window $(WINDOW_SIZE) -negative 25 -sample 1e-4 -threads $(NUM_THREADS) -binary 1 -iter $(MAX_ITER)

	@echo Evaluating Mikolov
	$(BUILDDIR)/compute_accuracy_mikolov $(SAVEFILE) < data/google_analogies_test_set/questions-words.txt
clean:
	rm -f $(BUILDDIR)/word2bits_mikolov
	rm -f $(BUILDDIR)/compute_accuracy_mikolov
	rm -f $(BUILDDIR)/glove
	rm -f $(BUILDDIR)/shuffle
	rm -f $(BUILDDIR)/cooccur
