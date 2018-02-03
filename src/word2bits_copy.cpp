#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include "word2bits.h"

using namespace std;

#define MAX_SENTENCE_LENGTH 1000

char *train_file;
double alpha = 0.05, starting_alpha;


/**
 * ======== TrainModelThread ========
 * This function performs the training of the model.
 */
void *TrainModelThread(void *id, Vocabulary *v) {

  /*
   * word - Stores the index of a word in the vocab table.
   */
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long sen[MAX_SENTENCE_LENGTH + 1];
  long long l2, c, target, label, local_iter = 1;
  unsigned long long next_random = (long long)id;
  double f, g;
  
  // neu1 is only used by the CBOW architecture.
  double *neu1 = (double *)calloc(SIZE, sizeof(double));
  
  // neu1e is used by both architectures.
  double *neu1e = (double *)calloc(SIZE, sizeof(double));
  
  
  // Open the training file and seek to the portion of the file that this 
  // thread is responsible for.
  ifstream in(train_file);
  int citer = 0;
  int words_processed = 0;

  clock_t start = clock();
  
  // This loop covers the whole training operation...
  while (1) {
    if (citer++ % 100000 == 0) {
      clock_t cur = clock();
      printf("Iteration %d %d (%lfk wps)\n", citer, local_iter,
	     words_processed / ((double)(cur-start+1) / (double)CLOCKS_PER_SEC * 1000));
      fflush(stdout);
    }
    // This 'if' block retrieves the next sentence from the training text and
    // stores it in 'sen'.
    // TODO - Under what condition would sentence_length not be zero?
    if (sentence_length == 0) {
      while (1) {
        // Read the next word from the training data and lookup its index in 
        // the vocab table. 'word' is the word's vocab index.
	string cur_word;
	in >> cur_word;
	word = v->word_to_index[cur_word];

	words_processed++;
	
        if (in.eof()) break;
        
        // If the word doesn't exist in the vocabulary, skip it.
        if (word == -1) continue;
        
        // 'vocab' word 0 is a special token "</s>" which indicates the end of 
        // a sentence.
        if (word == 0) break;

        if (SUBSAMPLING_COEFFICIENT > 0) {
          double ran = (sqrt((double)v->word_index_to_count[word] / (SUBSAMPLING_COEFFICIENT * v->n_total_words)) + 1) *
	    (SUBSAMPLING_COEFFICIENT * v->n_total_words) / (double)v->word_index_to_count[word];
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (double)65536) continue;
        }
        
        // If we kept the word, add it to the sentence.
        sen[sentence_length] = word;
        sentence_length++;
        
        // Verify the sentence isn't too long.
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      
      sentence_position = 0;
    }
    if (in.eof()) {
      local_iter--;
      if (local_iter == 0) break;
      in.close();
      in.open(train_file);
      sentence_length = 0;
      continue;
    }
    
    // Get the next word in the sentence. The word is represented by its index
    // into the vocab table.
    word = sen[sentence_position];
    
    if (word == -1) continue;
    
    for (c = 0; c < SIZE; c++) neu1[c] = 0;
    for (c = 0; c < SIZE; c++) neu1e[c] = 0;
    
    next_random = next_random * (unsigned long long)25214903917 + 11;
    
    // 'b' becomes a random integer between 0 and 'window'.
    // This is the amount we will shrink the WINDOW_SIZE size by.
    b = next_random % WINDOW_SIZE;
    
    // in -> hidden
    cw = 0;
    for (a = b; a < WINDOW_SIZE * 2 + 1 - b; a++) if (a != WINDOW_SIZE) {
        c = sentence_position - WINDOW_SIZE + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < SIZE; c++) neu1[c] += v->emb[c + last_word * SIZE];
        cw++;
      }
    if (cw) {
      for (c = 0; c < SIZE; c++) neu1[c] /= cw;
      for (d = 0; d < NEGATIVE_WINDOW_SIZE + 1; d++) {
	if (d == 0) {
	  target = word;
	  label = 1;
	} else {
	  next_random = next_random * (unsigned long long)25214903917 + 11;
	  target = v->unigram_table[(next_random >> 16) % v->unigram_table.size()];
	  if (target == 0) target = next_random % (v->n_unique_words - 1) + 1;
	  if (target == word) continue;
	  label = 0;
	}
	l2 = target * SIZE;
	f = 0;
	for (c = 0; c < SIZE; c++) f += neu1[c] * v->weights[c + l2];
	for (c = 0; c < SIZE; c++) v->weights[c + l2] += (label-f) * alpha * neu1[c];
      }
      // hidden -> in
      for (a = b; a < WINDOW_SIZE * 2 + 1 - b; a++) if (a != WINDOW_SIZE) {
          c = sentence_position - WINDOW_SIZE + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
	  for (c = 0; c < SIZE; c++) v->emb[c + last_word * SIZE] = v->weights[c + last_word * SIZE];	  
        }
    }
    
    
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  in.close();
  free(neu1);
  free(neu1e);
  return NULL;
}

/**
 * ======== TrainModel ========
 * Main entry point to the training process.
 */
void TrainModel(Vocabulary *v) {   

  TrainModelThread(0, v);
  
  FILE *fo = fopen("vectors.bin", "wb");
  // Save the word vectors
  fprintf(fo, "%d %d\n", v->n_unique_words, SIZE);
  for (int a = 0; a < v->n_unique_words; a++) {
    fprintf(fo, "%s ", v->index_to_word[a].c_str());
    for (int b = 0; b < SIZE; b++) fwrite(&v->emb[a * SIZE + b], sizeof(double), 1, fo);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

int main(int argc, char **argv) {  
  train_file = argv[1];
  Vocabulary *v = CreateVocabulary(train_file);
  TrainModel(v);
  return 0;
}
