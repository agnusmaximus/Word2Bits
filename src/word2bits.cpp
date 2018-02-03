#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include "word2bits.h"

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

/*
 * The size of the hash table for the vocabulary.
 * The vocabulary won't be allowed to grow beyond 70% of this number.
 * For instance, if the hash table has 30M entries, then the maximum
 * vocab size is 21M. This is to minimize the occurrence (and performance
 * impact) of hash collisions.
 */
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

/**
 * ======== vocab_word ========
 * Properties:
 *   cn - The word frequency (number of times it appears).
 *   word - The actual string word.
 */
struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

/*
 * ======== vocab ========
 * This array will hold all of the words in the vocabulary.
 * This is internal state.
 */
struct vocab_word *vocab;

int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

/*
 * ======== vocab_hash ========
 * This array is the hash table for the vocabulary. Word strings are hashed
 * to a hash code (an integer), then the hash code is used as the index into
 * 'vocab_hash', to retrieve the index of the word within the 'vocab' array.
 */
int *vocab_hash;

/*
 * ======== vocab_max_size ========
 * This is not a limit on the number of words in the vocabulary, but rather
 * a chunk size for allocating the vocabulary table. The vocabulary table will
 * be expanded as necessary, and is allocated, e.g., 1,000 words at a time.
 *
 * ======== vocab_size ========
 * Stores the number of unique words in the vocabulary. 
 * This is not a parameter, but rather internal state. 
 *
 * ======== layer1_size ========
 * This is the number of features in the word vectors.
 * It is the number of neurons in the hidden layer of the model.
 */
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/*
 *
 */
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/*
 * ======== alpha ========
 * TODO - This is a learning rate parameter.
 *
 * ======== starting_alpha ========
 *
 * ======== sample ========
 * This parameter controls the subsampling of frequent words.
 * Smaller values of 'sample' mean words are less likely to be kept.
 * Set 'sample' to 0 to disable subsampling.
 * See the comments in the subsampling section for more details.
 */
real alpha = 0.025, starting_alpha, sample = 1e-3;

real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

/**
 * ======== InitUnigramTable ========
 */
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    // [Chris] - The table may contain multiple elements which hold value 'i'.
    //           
    table[a] = i;
    // [Chris] - If the fraction of the table we have filled is greater than the
    //           fraction of this words weight / all word weights, then move to the next word.
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

/**
 * ======== ReadWord ========
 * Reads a single word from a file, assuming space + tab + EOL to be word 
 * boundaries.
 *
 * Parameters:
 *   word - A char array allocated to hold the maximum length string.
 *   fin  - The training file.
 */
void ReadWord(char *word, FILE *fin) {
  
  // 'a' will be the index into 'word'.
  int a = 0, ch;
  
  // Read until the end of the word or the end of the file.
  while (!feof(fin)) {
  
    // Get the next character.
    ch = fgetc(fin);
    
    // ASCII Character 13 is a carriage return 'CR' whereas character 10 is 
    // newline or line feed 'LF'.
    if (ch == 13) continue;
    
    // Check for word boundaries...
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      // If the word has at least one character, we're done.
      if (a > 0) {
        // Put the newline back before returning so that we find it next time.
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      // If the word is empty and the character is newline, treat this as the
      // end of a "sentence" and mark it with the token </s>.
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      // If the word is empty and the character is tab or space, just continue
      // on to the next character.     
      } else continue;
    }
    
    // If the character wasn't space, tab, CR, or newline, add it to the word.
    word[a] = ch;
    a++;
    
    // If the word's too long, truncate it, but keep going till we find the end
    // of it.
    if (a >= MAX_STRING - 1) a--;   
  }
  
  // Terminate the string with null.
  word[a] = 0;
}

/**
 * ======== GetWordHash ========
 * Returns hash value of a word. The hash is an integer between 0 and 
 * vocab_hash_size (default is 30E6).
 *
 * For example, the word 'hat':
 * hash = ((((h * 257) + a) * 257) + t) % 30E6
 */
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

/**
 * ======== SearchVocab ========
 * Lookup the index in the 'vocab' table of the given 'word'.
 * Returns -1 if the word is not found.
 * This function uses a hash table for fast lookup.
 */
int SearchVocab(char *word) {
  // Compute the hash value for 'word'.
  unsigned int hash = GetWordHash(word);
  
  // Lookup the index in the hash table, handling collisions as needed.
  // See 'AddWordToVocab' to see how collisions are handled.
  while (1) {
    // If the word isn't in the hash table, it's not in the vocab.
    if (vocab_hash[hash] == -1) return -1;
    
    // If the input word matches the word stored at the index, we're good,
    // return the index.
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    
    // Otherwise, we need to scan through the hash table until we find it.
    hash = (hash + 1) % vocab_hash_size;
  }
  
  // This will never be reached.
  return -1;
}

/**
 * ======== ReadWordIndex ========
 * Reads the next word from the training file, and returns its index into the
 * 'vocab' table.
 */
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

/**
 * ======== AddWordToVocab ========
 * Adds a new word to the vocabulary (one that hasn't been seen yet).
 */
int AddWordToVocab(char *word) {
  // Measure word length.
  unsigned int hash, length = strlen(word) + 1;
  
  // Limit string length (default limit is 100 characters).
  if (length > MAX_STRING) length = MAX_STRING;
  
  // Allocate and store the word string.
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  
  // Initialize the word frequency to 0.
  vocab[vocab_size].cn = 0;
  
  // Increment the vocabulary size.
  vocab_size++;
  
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  
  // Add the word to the 'vocab_hash' table so that we can map quickly from the
  // string to its vocab_word structure. 
  
  // Hash the word to an integer between 0 and 30E6.
  hash = GetWordHash(word);
  
  // If the spot is already taken in the hash table, find the next empty spot.
  while (vocab_hash[hash] != -1) 
    hash = (hash + 1) % vocab_hash_size;
  
  // Map the hash code to the index of the word in the 'vocab' array.  
  vocab_hash[hash] = vocab_size - 1;
  
  // Return the index of the word in the 'vocab' array.
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

/**
 * ======== SortVocab ========
 * Sorts the vocabulary by frequency using word counts, and removes words that
 * occur fewer than 'min_count' times in the training text.
 * 
 * Removing words from the vocabulary requires recomputing the hash table.
 */
void SortVocab() {
  int a, size;
  unsigned int hash;
  
  /*
   * Sort the vocabulary by number of occurrences, in descending order. 
   *
   * Keep </s> at the first position by sorting starting from index 1.
   *
   * Sorting the vocabulary this way causes the words with the fewest 
   * occurrences to be at the end of the vocabulary table. This will allow us
   * to free the memory associated with the words that get filtered out.
   */
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  
  // Clear the vocabulary hash table.
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // Store the initial vocab size to use in the for loop condition.
  size = vocab_size;
  
  // Recompute the number of training words.
  train_words = 0;
  
  // For every word currently in the vocab...
  for (a = 0; a < size; a++) {
    // If it occurs fewer than 'min_count' times, remove it from the vocabulary.
    if ((vocab[a].cn < min_count) && (a != 0)) {
      // Decrease the size of the new vocabulary.
      vocab_size--;
      
      // Free the memory associated with the word string.
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
   
  // Reallocate the vocab array, chopping off all of the low-frequency words at
  // the end of the table.
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

/**
 * ======== LearnVocabFromTrainFile ========
 * Builds a vocabulary from the words found in the training file.
 *
 * This function will also build a hash table which allows for fast lookup
 * from the word string to the corresponding vocab_word object.
 *
 * Words that occur fewer than 'min_count' times will be filtered out of
 * vocabulary.
 */
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  
  // Populate the vocab table with -1s.
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // Open the training file.
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  
  vocab_size = 0;
  
  // The special token </s> is used to mark the end of a sentence. In training,
  // the context window does not go beyond the ends of a sentence.
  // 
  // Add </s> explicitly here so that it occurs at position 0 in the vocab. 
  AddWordToVocab((char *)"</s>");
  
  while (1) {
    // Read the next word from the file into the string 'word'.
    ReadWord(word, fin);
    
    // Stop when we've reached the end of the file.
    if (feof(fin)) break;
    
    // Count the total number of tokens in the training text.
    train_words++;
    
    // Print progress at every 100,000 words.
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    
    // Look up this word in the vocab to see if we've already added it.
    i = SearchVocab(word);
    
    // If it's not in the vocab...
    if (i == -1) {
      // ...add it.
      a = AddWordToVocab(word);
      
      // Initialize the word frequency to 1.
      vocab[a].cn = 1;
    
    // If it's already in the vocab, just increment the word count.
    } else vocab[i].cn++;
    
    // If the vocabulary has grown too large, trim out the most infrequent 
    // words. The vocabulary is considered "too large" when it's filled more
    // than 70% of the hash table (this is to try and keep hash collisions
    // down).
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  
  // Sort the vocabulary in descending order by number of word occurrences.
  // Remove (and free the associated memory) for all the words that occur
  // fewer than 'min_count' times.
  SortVocab();
  
  // Report the final vocabulary size, and the total number of words 
  // (excluding those filtered from the vocabulary) in the training set.
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  
  file_size = ftell(fin);
  fclose(fin);
}

/**
 * ======== InitNet ========
 *
 */
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  
  // Allocate the hidden layer of the network, which is what becomes the word vectors.
  // The variable for this layer is 'syn0'.
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  // If we're using hierarchical softmax for training...
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  
  // If we're using negative sampling for training...
  if (negative>0) {
    // Allocate the output layer of the network. 
    // The variable for this layer is 'syn1neg'.
    // This layer has the same size as the hidden layer, but is the transpose.
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    // Set all of the weights in the output layer to 0.
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  
  // Randomly initialize the weights for the hidden layer (word vector layer).
  // TODO - What's the equation here?
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }  
}

/**
 * ======== TrainModelThread ========
 * This function performs the training of the model.
 */
void *TrainModelThread(void *id) {

  /*
   * word - Stores the index of a word in the vocab table.
   * word_count - Stores the total number of training words processed.
   */
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  
  // neu1 is only used by the CBOW architecture.
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  
  // neu1e is used by both architectures.
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  
  
  // Open the training file and seek to the portion of the file that this 
  // thread is responsible for.
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  
  // This loop covers the whole training operation...
  while (1) {
    // This block prints a progress update, and also adjusts the training 
    // 'alpha' parameter.
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    
    // This 'if' block retrieves the next sentence from the training text and
    // stores it in 'sen'.
    // TODO - Under what condition would sentence_length not be zero?
    if (sentence_length == 0) {
      while (1) {
        // Read the next word from the training data and lookup its index in 
        // the vocab table. 'word' is the word's vocab index.
        word = ReadWordIndex(fi);
        
        if (feof(fi)) break;
        
        // If the word doesn't exist in the vocabulary, skip it.
        if (word == -1) continue;
        
        // Track the total number of training words processed.
        word_count++;
        
        // 'vocab' word 0 is a special token "</s>" which indicates the end of 
        // a sentence.
        if (word == 0) break;
        
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;         
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        
        // If we kept the word, add it to the sentence.
        sen[sentence_length] = word;
        sentence_length++;
        
        // Verify the sentence isn't too long.
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    
    // Get the next word in the sentence. The word is represented by its index
    // into the vocab table.
    word = sen[sentence_position];
    
    if (word == -1) continue;
    
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    
    next_random = next_random * (unsigned long long)25214903917 + 11;
    
    // 'b' becomes a random integer between 0 and 'window'.
    // This is the amount we will shrink the window size by.
    b = next_random % window;
    
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
	  for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
	  //for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
	  for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += (label-f) * alpha * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          //for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
	  for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] = syn1neg[c + last_word * layer1_size];	  
        }
      }
    } 
    
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
}

/**
 * ======== TrainModel ========
 * Main entry point to the training process.
 */
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
    
  printf("Starting training using file %s\n", train_file);
  
  starting_alpha = alpha;  
  LearnVocabFromTrainFile();
    
  // Allocate the weight matrices and initialize them.
  InitNet();

  // If we're using negative sampling, initialize the unigram table, which
  // is used to pick words to use as "negative samples" (with more frequent
  // words being picked more often).  
  if (negative > 0) InitUnigramTable();
  
  // Record the start time of training.
  start = clock();

  TrainModelThread((void *)a);
  
  fo = fopen("vectors.bin", "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary)
	for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } 
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {  
  // Allocate the vocabulary table.
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));  
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  TrainModel();
  return 0;
}
