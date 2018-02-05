CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall -lpthread 

benchmark:
	$(CC) -w src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
	$(CC) src/word2bits.cpp $(CFLAGS) -o word2bits
	./word2bits -train /afs/ir.stanford.edu/users/m/a/maxlam//text8 -output vectors.bin -size 400 -window 8 -negative 25 -sample 1e-4 -threads 15 -binary 1 -iter 40
	./compute_accuracy vectors.bin < data/google_analogies_test_set/questions-words.txt
accuracy:
	$(CC) src/compute-accuracy.c $(CFLAGS) -o compute_accuracy
word2vec:
	$(CC) src/word2vec.c $(CFLAGS) -o word2vec
clean:
	rm -f word2bits
	rm -f compute_accuracy
