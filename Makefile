CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall -lpthread

# Data file path
TEXT8_PATH := /afs/ir.stanford.edu/users/m/a/maxlam//text8

# Path to source code
MIKOLOV_DIR := ./src/mikolov/
GLOVE_DIR := ./src/glove/src/

# Path to binary outputs
BUILDDIR := ./build/

# Path to saved vector outputs
SAVEDIR := ./save_vectors/

benchmark-glove:
	$(CC) $(GLOVE_DIR)/glove.c -o $(BUILDDIR)/glove $(CFLAGS)
	$(CC) $(GLOVE_DIR)/shuffle.c -o $(BUILDDIR)/shuffle $(CFLAGS)
	$(CC) $(SRCDIR)/cooccur.c -o $(BUILDDIR)/cooccur $(CFLAGS)

benchmark-mikolov:
	$(CC) -w  $(MIKOLOV_DIR)/compute-accuracy.c $(CFLAGS) -o $(BUILDDIR)/compute_accuracy_mikolov
	$(CC) $(MIKOLOV_DIR)/word2bits.cpp $(CFLAGS) -o $(BUILDDIR)/word2bits_mikolov
	$(BUILDDIR)/word2bits_mikolov -train $(TEXT8_PATH) -output $(SAVEDIR)/mikolov-vectors.bin -size 300 -window 8 -negative 25 -sample 1e-4 -threads 10 -binary 1 -iter 1
	$(BUILDDIR)/compute_accuracy_mikolov $(SAVEDIR)/mikolov-vectors.bin < data/google_analogies_test_set/questions-words.txt
clean:
	rm -f $(BUILDDIR)/word2bits_mikolov
	rm -f $(BUILDDIR)/compute_accuracy_mikolov
	rm -f $(BUILDDIR)/glove
	rm -f $(BUILDDIR)/shuffle
	rm -f $(BUILDDIR)/cooccur
