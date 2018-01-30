CC=g++
CFLAGS=-Ofast -march=native

word2bits:
	$(CC) src/word2bits.cpp $(CFLAGS) -o word2bits
clean:
	rm -f word2bits
