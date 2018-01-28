#include <iostream>
#include <ctime>
#include <cstdio>
#include <stdlib.h>
#include <assert.h>
#include "sig_attr.h"

using namespace std;

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

    clock_t start = clock();
    int out_sig_attr, out_sig_attr_row;
    for (int i = 0; i < n_reps; i++) {
	sig_attr_general(bits, n_cols, n_rows,
			 &out_sig_attr,
			 &out_sig_attr_row);
    }
    double duration = (clock() - start) / (double)CLOCKS_PER_SEC;

    cout << "----------------------------------------------" << endl;
    cout << "Sigattr " << n_rows << "x" << n_cols << endl;
    cout << "----------------------------------------------" << endl;
    cout << n_reps << " calculates sig attr in " << duration << " seconds. " << duration/n_reps << " seconds per sig attr." << endl;
    cout << "----------------------------------------------" << endl;
}

int main() {
    cout << "Starting sig attr calculation benchmark..." << endl;
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
