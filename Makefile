CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall

accuracy:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
word2vec:
	$(CC) src/word2vec.c $(CFLAGS) -o word2vec
clean:
	rm -f word2bits
