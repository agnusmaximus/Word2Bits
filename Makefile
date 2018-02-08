CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall -lpthread 

benchmark-mikolov:
	$(CC) -w src/mikolov/compute-accuracy.c $(CFLAGS) -o compute_accuracy_mikolov
	$(CC) src/mikolov/word2bits.cpp $(CFLAGS) -o word2bits_mikolov
	./word2bits_mikolov -train /afs/ir.stanford.edu/users/m/a/maxlam//text8 -output mikolov-vectors.bin -size 300 -window 8 -negative 25 -sample 1e-4 -threads 10 -binary 1 -iter 1
	./compute_accuracy_mikolov mikolov-vectors.bin < data/google_analogies_test_set/questions-words.txt
clean:
	rm -f word2bits_mikolov
	rm -f compute_accuracy_mikolov
