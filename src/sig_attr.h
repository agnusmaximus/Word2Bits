#include <iostream>
#include <assert.h>
#include "bit_transpose.h"

void sig_attr_general(char *bit_matrix,
		      int vector_bit_width,
		      int n_vectors,
		      int *out_sig_attr, int *out_sig_attr_row) {
    assert(vector_bit_width % 8 == 0);
    assert(n_vectors % 8 == 0);
    unsigned char transposed_bit_matrix[n_vectors * vector_bit_width/8];
    ssebmx_m(bit_matrix, (char *)transposed_bit_matrix, n_vectors, vector_bit_width);
    int max_count = 0;
    for (int i = 0; i < vector_bit_width; i++) {
	int count = 0;
	for (int j = 0; j < n_vectors/8; j++)
	    count += __builtin_popcount(transposed_bit_matrix[i*n_vectors/8+j]);
	if (count > max_count) {
	    *out_sig_attr = 1;
	    *out_sig_attr_row = i;
	    max_count = count;
	}
	else if (n_vectors-count > max_count) {
	    *out_sig_attr = 0;
	    *out_sig_attr_row = i;
	    max_count = n_vectors-count;
	}
    }
}

void sig_attr_window32(char *bit_matrix,
		       int vector_bit_width,
		       int n_vectors,
		       int *out_sig_attr, int *out_sig_attr_row) {
    assert(vector_bit_width % 8 == 0);
    assert(n_vectors % 32 == 0);
    unsigned char transposed_bit_matrix[n_vectors * vector_bit_width/8];
    unsigned int *transposed_bit_matrix_ptr = (unsigned int *)transposed_bit_matrix;
    ssebmx_m(bit_matrix, (char *)transposed_bit_matrix, n_vectors, vector_bit_width);
    int max_count = 0;
    for (int i = 0; i < vector_bit_width; i++) {
	int count = 0;
	for (int j = 0; j < n_vectors/32; j++)
	    count += __builtin_popcount(transposed_bit_matrix_ptr[i*n_vectors/32+j]);
	if (count > max_count) {
	    *out_sig_attr = 1;
	    *out_sig_attr_row = i;
	    max_count = count;
	}
	if (n_vectors-count > max_count) {
	    *out_sig_attr = 0;
	    *out_sig_attr_row = i;
	    max_count = n_vectors-count;
	}
    }
}
