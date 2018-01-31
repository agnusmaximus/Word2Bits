CC=g++
CFLAGS=-Ofast -march=native -std=c++11 -Wall -pg -fprofile-arcs -ftest-coverage 

word2bits:
	$(CC) src/word2bits.cpp $(CFLAGS) -o word2bits -DBMI_ENABLED
clean:
	rm -f word2bits
