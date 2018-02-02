CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall

benchmark:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
	$(CC) src/word2vec.c $(CFLAGS) -o word2vec
	./word2vec -train ~/text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
	./word-analogy vectors.bin

accuracy:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
word2vec:
	$(CC) src/word2vec.c $(CFLAGS) -o word2vec
clean:
	rm -f word2vec
	rm -f compute_accuracy
