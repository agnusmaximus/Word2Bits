CC=g++
CFLAGS=-Ofast -march=native -std=c++11

word2bits:
	$(CC) src/word2bits.cpp $(CFLAGS) -o word2bits
clean:
	rm -f word2bits
