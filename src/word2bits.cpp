#define NDEBUG

#include <iostream>
#include <math.h>
#include <vector>
#include <immintrin.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <map>
#include <ctime>
#include <time.h>
#include "word2bits.h"
#include "evaluate.h"

using namespace std;

// Worker train function
void TrainWorker(const char *filepath, int id, Vocabulary *vocab) {

    printf("Worker %d beginning training...\n", id);

    // Workers process file in parallel (but in different starting positions)
    unsigned long seek_pos = (id / N_WORKERS) * vocab->filesize;
    ifstream fin(filepath);
    fin.seekg(seek_pos);

    // Embeddings 
    char *emb1;
    emb1 = vocab->emb1;

    // Get rid of first string (which might've been cut off)
    string word;
    fin >> word;

    // Create contexts
    Context *positive_context, *negative_context;
    positive_context = CreateContext(vocab, WINDOW_SIZE);
    negative_context = CreateContext(vocab, NEGATIVE_WINDOW_SIZE);

    // Populate positive context
    for (int i = 0; i < WINDOW_SIZE; i++) {
	fin >> word;
	int word_index = WordToIndex(vocab, word);
	char *bits = WordToBits(vocab, word_index, emb1);
	AddWordToContext(positive_context, word_index, bits);
    }

    // Populate negative context
    for (int i = 0; i < NEGATIVE_WINDOW_SIZE; i++) {
	int word_index = NegativeSample(vocab);
	char *bits = WordToBits(vocab, word_index, emb1);
	AddWordToContext(negative_context, word_index, bits);
    }

    int words_processed = 0;
    int n_epochs_processed = 0;
    float negative_running_loss = 0;
    time_t tstart = time(0);
    while (n_epochs_processed != N_EPOCHS_PER_WORKER) {
      	if (words_processed++ % PRINT_INTERVAL == 0) {
	    time_t tcur = time(0);
	    float elapsed = difftime(tcur, tstart);
	    if (elapsed >= 0) {
		printf("- %f words/sec (epoch %d) (neg loss %lf)\n",
		       words_processed/elapsed, n_epochs_processed+1,
		       negative_running_loss / PRINT_INTERVAL / BITSIZE);		
	    }
	    
	    negative_running_loss = 0;

	    int i1 = vocab->word_to_index["one"];
	    int i2 = vocab->word_to_index["two"];
	    int i3 = vocab->word_to_index["three"];
	    int i4 = vocab->word_to_index["dog"];
	    PrintVectorDifference(vocab, i1, i2);
	    PrintVectorDifference(vocab, i1, i3);
	    PrintVectorDifference(vocab, i2, i3);
	    PrintVectorDifference(vocab, i1, i4);
	    printf("--\n");
	    int i5 = vocab->word_to_index["england"];
	    int i6 = vocab->word_to_index["english"];
	    int i7 = vocab->word_to_index["japan"];
	    int i8 = vocab->word_to_index["japanese"];
	    int i9 = vocab->word_to_index["israel"];
	    int i10 = vocab->word_to_index["israeli"];
	    PrintVectorDifference(vocab, i5, i6);
	    PrintVectorDifference(vocab, i7, i8);
	    PrintVectorDifference(vocab, i9, i10);
	    //int print_word_index = center_word_index;
	    //int print_word_index = vocab->word_to_index["man"];
	    //printf("%s: \n", vocab->index_to_word[print_word_index].c_str());
	    //PrintBits(WordToBits(vocab, print_word_index, emb1));
	    //print_word_index = vocab->word_to_index["woman"];
	    //printf("%s: \n", vocab->index_to_word[print_word_index].c_str());
	    //PrintBits(WordToBits(vocab, print_word_index, emb1));
	    //print_word_index = vocab->word_to_index["king"];
	    //printf("%s: \n", vocab->index_to_word[print_word_index].c_str());
	    //PrintBits(WordToBits(vocab, print_word_index, emb1));
	    //print_word_index = vocab->word_to_index["queen"];
	    //printf("%s: \n", vocab->index_to_word[print_word_index].c_str());
	    //PrintBits(WordToBits(vocab, print_word_index, emb1));
	}

	// Reset file pointer if read to end.
	if (fin.eof()) {
	  fin.close();
	  fin.open(filepath);
	  n_epochs_processed++;

	  if (n_epochs_processed % 2 == 0)
	    evaluate_google_analogies(vocab);
	}

	// Update center word in positive context
	int center_word_index = CenterWordOfContext(positive_context);
	for (int i = 0; i < BITSIZE; i += BITS_PER_BYTE) {
	  char cur_byte = CurrentWordBits(positive_context)[i/BITS_PER_BYTE];
	  unsigned char mask = 1 << (BITS_PER_BYTE-1);
	  for (char k = 0; k < BITS_PER_BYTE; k++) {
	    
	    // Calculate counts of 1s for this bit
	    int self_count = (cur_byte & mask) != 0;
	    mask >>= 1;
	    int positive_counts = positive_context->bitcounts[i+k];
	    int negative_counts = negative_context->bitcounts[i+k];
	    float weight = positive_counts/(float)WINDOW_SIZE - self_count/(float)WINDOW_SIZE
	      + (NEGATIVE_WINDOW_SIZE-negative_counts)/(float)NEGATIVE_WINDOW_SIZE;
	    
	    // Calculate loss
	    float neg_loss = 0;
#ifndef NDEBUG
	    neg_loss =
	      ((self_count * positive_counts) +
	       (1-self_count) * (WINDOW_SIZE-positive_counts)) / (float)WINDOW_SIZE +
	      ((self_count * (NEGATIVE_WINDOW_SIZE - negative_counts)) +
	       (1-self_count) * (negative_counts)) / (float)NEGATIVE_WINDOW_SIZE;
	       negative_running_loss += neg_loss;
#endif

	    // Divide weight by 2 (half of the gradient comes from positive context, half from negative)
	       //float norm_weight = weight / 2;
            float norm_weight = weight / 2;
	    float grad = norm_weight - .5;
	    vocab->weights[center_word_index*BITSIZE + i+k] += LEARNING_RATE * grad;
	    //vocab->weights[center_word_index*BITSIZE + i+k] += (LEARNING_RATE/10 * ((float)fast_rand() / (FAST_RAND_MAX) - .5));
	    //vocab->weights[center_word_index*BITSIZE + i+k] = min((float)1, max((float)0,vocab->weights[center_word_index*BITSIZE + i+k]));	    
	    // Assertions
	    assert(positive_counts >= self_count);
	    assert(negative_counts <= NEGATIVE_WINDOW_SIZE);
	    assert(positive_counts <= WINDOW_SIZE);
	    assert(negative_counts >= 0);
	    assert(positive_counts >= 0);
	    assert(self_count == 0 || self_count == 1);
	    assert(weight >= 0);
	    assert(NEGATIVE_WINDOW_SIZE >= negative_counts);
	    assert(positive_counts <= WINDOW_SIZE);
	    assert(neg_loss <= (WINDOW_SIZE + NEGATIVE_WINDOW_SIZE));
	    assert(weight < (WINDOW_SIZE+NEGATIVE_WINDOW_SIZE));
	    assert(norm_weight >= 0 && norm_weight <= 1);
	  }
	}

	// Write old word context to memory.
	UpdateBits(vocab, center_word_index);
	
	// Add Word to context for both positive and negative contexts w/ subsampling.
	int positive_word_index = -1;
	while (!Keep(vocab, positive_word_index)) {
	    if (fin.eof()) {
	      fin.close();
	      fin.open(filepath);
	    }
	    fin >> word;
	    positive_word_index = WordToIndex(vocab, word);
	}
	char *positive_bits = WordToBits(vocab, positive_word_index, emb1);
	AddWordToContext(positive_context, positive_word_index, positive_bits);
	
	int negative_word_index = -1;
	while (!Keep(vocab, negative_word_index)) {
	    negative_word_index = NegativeSample(vocab);
	}
	char *negative_bits = WordToBits(vocab, negative_word_index, emb1);
	AddWordToContext(negative_context, negative_word_index, negative_bits);
    }

    fin.close();
    DestroyContext(positive_context);
    DestroyContext(negative_context);

    printf("Worker %d done training.\n", id);
}

// Main entry point for training.
void Train(const char *filepath) {

    // Build vocabulary
    Vocabulary *vocab = CreateVocabulary(filepath);

    // Train
    TrainWorker(filepath, 0, vocab);

    DestroyVocabulary(vocab);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    if (argc != 2) {
	printf("Usage: ./word2bits input_file_path\n");
	exit(0);
    }
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
	printf("Unable to find file: %s\n", argv[1]);
	exit(0);
    }
    printf("------------------------------- ");
    printf("Word2bits: %s", argv[1]);
    printf(" -------------------------------\n");
    Train(argv[1]);
}
