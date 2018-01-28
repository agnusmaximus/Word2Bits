#include <iostream>
#include <ctime>
#include <cstdio>
#include <stdlib.h>
#include <assert.h>
#include "../base/bit_transpose.h"

using namespace std;

const char *byte_to_binary(int x) {
    static char b[9];
    b[0] = '\0';
    int z;
    for (z = 128; z > 0; z >>= 1)
        strcat(b, ((x & z) == z) ? "1" : "0");
    return b;
}

void print_bit_matrix(char *matrix, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
	for (int j = 0; j < n_cols; j++) {
	    cout << byte_to_binary(matrix[i*n_cols+j]);
	}
	cout << endl;
    }
}

void benchmark_bit_transpose(int n_rows, int n_cols, int n_reps) {
    if (n_rows % 8 != 0 || n_cols % 8 != 0) {
	cout << "n_rows and n_cols must be divisible by 8." << endl;
	exit(0);
    }
    if (n_cols < 8 || n_cols % 8 != 0) {
	cout << "n_cols (vector width) must be divisible by 8." << endl;
	exit(0);
    }

    // Random bit pattern
    int n_cols_effective = n_cols / 8;
    char bits[n_rows*n_cols_effective];
    for (int i = 0; i < n_rows; i++) {
	for (int j = 0; j < n_cols_effective; j++) {
	    bits[i*n_cols_effective+j] = rand() % 255;
	}
    }

    // Transpose
    char transposed_bits[n_rows*n_cols_effective];

    clock_t start = clock();
    for (int i = 0; i < n_reps; i++) {
	ssebmx_m(bits, transposed_bits, n_rows, n_cols);
    }
    double duration = (clock() - start) / (double)CLOCKS_PER_SEC;

    cout << "----------------------------------------------" << endl;
    cout << "Transpose " << n_rows << "x" << n_cols << endl;
    cout << "----------------------------------------------" << endl;
    cout << n_reps << " transposes in " << duration << " seconds. " << duration/n_reps << " seconds per transpose." << endl;
    cout << "----------------------------------------------" << endl;
}

void check_correctness_bit_transpose(int n_rows, int n_cols) {
    if (n_rows % 8 != 0 || n_cols % 8 != 0) {
	cout << "n_rows and n_cols must be divisible by 8." << endl;
	exit(0);
    }
    if (n_cols < 8 || n_cols % 8 != 0) {
	cout << "n_cols (vector width) must be divisible by 8." << endl;
	exit(0);
    }

    // Random bit pattern
    int n_cols_effective = n_cols / 8;
    char bits[n_rows*n_cols_effective];
    for (int i = 0; i < n_rows; i++) {
	for (int j = 0; j < n_cols_effective; j++) {
	    bits[i*n_cols_effective+j] = rand() % 255;
	}
    }

    // Transpose
    char transposed_bits[n_rows*n_cols_effective];
    ssebmx_m(bits, transposed_bits, n_rows, n_cols);
    char transposed_transposed_bits[n_rows*n_cols_effective];
    ssebmx_m(transposed_bits,
	     transposed_transposed_bits,
	     n_cols, n_rows);

    // Check transpose of transpose is same
    assert(memcmp(bits, transposed_transposed_bits,
		  sizeof(char) * n_rows*n_cols_effective) == 0);
}

int main() {
    cout << "Checking correctness of bit transpose..." << endl;
    check_correctness_bit_transpose(8, 64);
    check_correctness_bit_transpose(8, 256);
    check_correctness_bit_transpose(8, 1024);
    cout << "Starting bit transpose benchmark..." << endl;
    benchmark_bit_transpose(8, 8, 1000000);
    benchmark_bit_transpose(8, 16, 1000000);
    benchmark_bit_transpose(8, 64, 1000000);
    benchmark_bit_transpose(8, 256, 1000000);
    benchmark_bit_transpose(8, 512, 1000000);
    benchmark_bit_transpose(8, 1024, 1000000);
    benchmark_bit_transpose(16, 64, 1000000);
    benchmark_bit_transpose(16, 256, 1000000);
    benchmark_bit_transpose(16, 512, 1000000);
    benchmark_bit_transpose(16, 1024, 1000000);
    cout << "Done." << endl;
}
