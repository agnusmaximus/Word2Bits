CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall

benchmark_bits:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
	$(CC) src/word2bits_copy.cpp $(CFLAGS) -o word2bits
	./word2bits ~/text8
	./compute_accuracy vectors.bin < data/google_analogies_test_set/questions-words.txt

benchmark:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
	$(CC) src/word2vec.c $(CFLAGS) -o word2vec
	./word2vec -train ~/text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 15
	./compute_accuracy vectors.bin < data/google_analogies_test_set/questions-words.txt
accuracy:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
word2vec:
	$(CC) src/word2vec.c $(CFLAGS) -o word2vec
clean:
	rm -f word2vec
	rm -f compute_accuracy
