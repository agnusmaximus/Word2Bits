CC_MIKOLOV=g++
CFLAGS=-O3 -march=native -lm -pthread -Wno-unused-result

word2bits:
	$(CC_MIKOLOV) $(CFLAGS) ./src/word2bits.cpp -o word2bits
compute_accuracy:
	$(CC_MIKOLOV) $(CFLAGS) ./src/compute-accuracy.c -o compute_accuracy
clean:
	rm -f word2bits
	rm -f compute_accuracy
