#include <iostream>
#include <ctime>
#include <cstdio>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../base/sig_attr.h"

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

void check_correctness_sig_attr(int n_rows, int n_cols) {
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

    // Compute correct answer
    int counts[n_cols];
    memset(counts, 0, sizeof(int) * n_cols);
    for (int i = 0; i < n_rows; i++) {
	for (int j = 0; j < n_cols_effective; j++) {
	    for (int k = 7; k >= 0; k--) {
		int is_set = (bits[i*n_cols_effective+j] & (1 << k)) != 0;
		counts[j*8 + (7-k)] += is_set;
	    }
	}
    }
    int correct_sig_attr = -1, correct_sig_attr_row = -1, max_count = 0;
    for (int i = 0; i < n_cols; i++) {
	if (counts[i] > max_count) {
	    max_count = counts[i];
	    correct_sig_attr = 1;
	    correct_sig_attr_row = i;
	}
	if (n_rows-counts[i] > max_count) {
	    max_count = n_rows-counts[i];
	    correct_sig_attr = 0;
	    correct_sig_attr_row = i;
	}
    }

    // Call sig attr
    int sig_attr, sig_attr_row;
    sig_attr_general(bits, n_cols, n_rows,
		     &sig_attr, &sig_attr_row);

    printf("Got: %d vs Expected: %d\n",
	   sig_attr, correct_sig_attr);
    printf("Got: %d vs Expected: %d\n",
	   sig_attr_row, correct_sig_attr_row);
    assert(sig_attr == correct_sig_attr);
    assert(sig_attr_row == correct_sig_attr_row);
}

void benchmark_sig_attr(int n_rows, int n_cols, int n_reps) {
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

void benchmark_sig_attr_window32(int n_rows, int n_cols, int n_reps) {
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
	sig_attr_window32(bits, n_cols, n_rows,
			  &out_sig_attr,
			  &out_sig_attr_row);
    }
    double duration = (clock() - start) / (double)CLOCKS_PER_SEC;

    cout << "----------------------------------------------" << endl;
    cout << "Sigattr window32 " << n_rows << "x" << n_cols << endl;
    cout << "----------------------------------------------" << endl;
    cout << n_reps << " calculates sig attr in " << duration << " seconds. " << duration/n_reps << " seconds per sig attr window32." << endl;
    cout << "----------------------------------------------" << endl;
}

int main() {
    cout << "Checking correctness of bit sig_attr..." << endl;
    check_correctness_sig_attr(8, 8);
    check_correctness_sig_attr(8, 64);
    check_correctness_sig_attr(8, 256);
    check_correctness_sig_attr(8, 1024);
    check_correctness_sig_attr(16, 1024);
    check_correctness_sig_attr(32, 1024);
    check_correctness_sig_attr(64, 1024);
    cout << "Starting sig attr calculation benchmark..." << endl;
    benchmark_sig_attr(8, 8, 1000000);
    benchmark_sig_attr(8, 16, 1000000);
    benchmark_sig_attr(8, 64, 1000000);
    benchmark_sig_attr(8, 256, 1000000);
    benchmark_sig_attr(8, 512, 1000000);
    benchmark_sig_attr(8, 1024, 1000000);
    benchmark_sig_attr(16, 64, 1000000);
    benchmark_sig_attr(16, 256, 1000000);
    benchmark_sig_attr(16, 512, 1000000);
    benchmark_sig_attr(16, 1024, 1000000);
    benchmark_sig_attr(32, 512, 1000000);
    benchmark_sig_attr_window32(32, 512, 1000000);
    cout << "Done." << endl;
}
