CC=g++
CFLAGS=-Ofast

benchmark_bit_transpose:
	rm -f bit_transpose_benchmark
	$(CC) src/benchmark/benchmark_bit_transpose.cpp $(CFLAGS) -o bit_transpose_benchmark
	./bit_transpose_benchmark

benchmark_sig_attr:
	rm -f sig_attr_benchmark
	$(CC) src/benchmark/benchmark_sig_attr.cpp $(CFLAGS) -o sig_attr_benchmark
	./sig_attr_benchmark

clean:
	rm bit_transpose_benchmark
	rm sig_attr_benchmark
