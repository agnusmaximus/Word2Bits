//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


#include <iostream>
#include <stdio.h>
#include <limits>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <pthread.h>

using namespace std;
typedef numeric_limits< double > dbl;

#define MAX_STRING 4096
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1, bitlevel = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
bool save_every_epoch = 0;
real alpha = 0.05, starting_alpha, sample = 1e-3;
real reg = 0;
double *thread_losses;
real *u, *v, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

///////////////////////////////////////////
//              Word2Bits                //
///////////////////////////////////////////

real sigmoid(real val) {
  if (val > MAX_EXP) return 1;
  if (val < -MAX_EXP) return 1e-9;
  return 1 / (1 + (real)exp(-val));
}

real quantize(real num, int bitlevel) {

  if (bitlevel == 0) {
    // Special bitlevel 0 => full precision
    return num;
  }
  
  // Extract sign
  real retval = 0;
  real sign = num < 0 ? -1 : 1;
  num *= sign;
  
  // Boundaries: 0
  if (bitlevel == 1) {
    return sign / 3;
  }
  
  // Determine boundary and discrete activation value (2 bits)
  // Boundaries: 0, .5
  if (bitlevel == 2) {
    if (num >= 0 && num <= .5) retval = .25; 
    else retval = .75;
  }

  // Determine boundary and discrete activation value (4 bits = 16 values)
  // Boundaries: 0, .1, .2, .3, .4, .5, .6, .7, .8
  //real boundaries[] = {0, .25, .5, .75, 1, 1.25, 1.5, 1.75};
  if (bitlevel >= 4) {
    int segmentation = pow(2, bitlevel-1);
    int casted = (num * segmentation) + (real).5;
    casted = casted > segmentation ? segmentation : casted;
    retval = casted / (real)segmentation;
  }

  return sign * retval;
}

/////////////////////////////////////////////////////////////

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = fgetc(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
	if (ch == '\n') ungetc(ch, fin);
	break;
      }
      if (ch == '\n') {
	strcpy(word, (char *)"</s>");
	return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
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

void LearnVocabFromTrainFile() {
  char word[MAX_STRING], eof = 0;
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    train_words++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    if (fscanf(fin, "%lld%c", &vocab[a].cn, &c));
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&u, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (u == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&v, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (v == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < vocab_size; a++) {
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      v[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) ;
    }
  }
  for (a = 0; a < vocab_size; a++) 
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      u[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) ;
    }
}

void *TrainModelThread(void *id) {
  long long a, b, c, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  // local_iter = 1 to save the model every epoch
  long long l2, target, label, local_iter = 1;
  unsigned long long next_random = (long long)id;
  char eof = 0;
  int local_bitlevel = bitlevel;
  real f, g;
  clock_t now;
  real *context_avg = (real *)calloc(layer1_size, sizeof(real));
  real *context_avge = (real *)calloc(layer1_size, sizeof(real));
  double loss = 0, total_loss = 0;  
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
	now=clock();
	printf("%cAlpha: %f  Progress: %.2f%%  Cost: %f Words/thread/sec: %.2fk  ", 13, alpha,
	       word_count_actual / (real)(iter * train_words + 1) * 100,
	       loss,
	       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
	loss = 0;
	fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
	word = ReadWordIndex(fi, &eof);
	if (eof) break;
	if (word == -1) continue;
	word_count++;
	if (word == 0) break;
	// The subsampling randomly discards frequent words while keeping the ranking same
	if (sample > 0) {
	  real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) *
	    (sample * train_words) / vocab[word].cn;
	  next_random = next_random * (unsigned long long)25214903917 + 11;
	  if (ran < (next_random & 0xFFFF) / (real)65536) continue;
	}
	sen[sentence_length] = word;
	sentence_length++;
	if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) context_avg[c] = 0;
    for (c = 0; c < layer1_size; c++) context_avge[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    cw = 0;
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
	c = sentence_position - window + a;
	if (c < 0) continue;
	if (c >= sentence_length) continue;
	last_word = sen[c];
	if (last_word == -1) continue;
	real local_reg_loss = 0;
	for (c = 0; c < layer1_size; c++) {
	  real cur_val = quantize(u[c + last_word * layer1_size], local_bitlevel);
	  context_avg[c] += cur_val;
	  local_reg_loss += cur_val * cur_val;
	}
	local_reg_loss = reg * local_reg_loss;
	loss += -local_reg_loss;
	total_loss += -local_reg_loss;
	cw++;
      }
    if (cw) {
      for (c = 0; c < layer1_size; c++) context_avg[c] /= cw;
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
	real local_reg_loss = 0;
	for (c = 0; c < layer1_size; c++) {
	  real cur_val = quantize(v[c + l2], local_bitlevel);
	  f += context_avg[c] * cur_val;

	  // Keep track of regularization loss
	  local_reg_loss += cur_val * cur_val;
	}
	local_reg_loss = reg * local_reg_loss;
	
	if (f > MAX_EXP) g = (label - 1) * alpha;
	else if (f < -MAX_EXP) g = (label - 0) * alpha;
	else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

	////////////////////
	// Compute loss
	////////////////////
	real dot_product = f * pow(-1, 1-label);
	real local_loss = log(sigmoid(dot_product));
	loss += local_loss - local_reg_loss;
	total_loss += local_loss - local_reg_loss;
	/////////////////////
	
	for (c = 0; c < layer1_size; c++) {
	  context_avge[c] += g * quantize(v[c + l2], local_bitlevel);
	}
	for (c = 0; c < layer1_size; c++) {
	  v[c + l2] += g * context_avg[c] - 2*alpha*reg*v[c + l2];
	}
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
	  c = sentence_position - window + a;
	  if (c < 0) continue;
	  if (c >= sentence_length) continue;
	  last_word = sen[c];
	  if (last_word == -1) continue;
	  for (c = 0; c < layer1_size; c++) {
	    u[c + last_word * layer1_size] += context_avge[c] - 2*alpha*reg*u[c+last_word*layer1_size];
	  }
	}
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  thread_losses[(long long)id] = total_loss;
  fclose(fi);
  free(context_avg);
  free(context_avge);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fo;
  thread_losses = (double *)malloc(sizeof(double) * num_threads);
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();

  start = clock();
  for (int iteration = 0; iteration < iter; iteration++) {
    printf("Starting epoch: %d\n", iteration);
    memset(thread_losses, 0, sizeof(double) * num_threads);
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    double total_loss_epoch = 0;
    for (a = 0; a < num_threads; a++) total_loss_epoch += thread_losses[a];
    printf("Epoch Loss: %lf\n", total_loss_epoch);
    char output_file_cur_iter[MAX_STRING] = {0};
    sprintf(output_file_cur_iter, "%s_epoch%d", output_file, iteration);
    if (classes == 0 && save_every_epoch) {
      fo = fopen(output_file_cur_iter, "wb");
      // Save the word vectors
      fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
      for (a = 0; a < vocab_size; a++) {
	fprintf(fo, "%s ", vocab[a].word);
	for (b = 0; b < layer1_size; b++) {
	  float avg = u[a*layer1_size+b] + v[a*layer1_size+b];
	  avg = quantize(avg, bitlevel);
	  if (binary) fwrite(&avg, sizeof(float), 1, fo);
	  else fprintf(fo, "%lf ", avg);
	}
	fprintf(fo, "\n");
      }
      fclose(fo);
    }
  }

  // Write an extra file
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (b = 0; b < layer1_size; b++) {
	float avg = u[a*layer1_size+b] + v[a*layer1_size+b];
	avg = quantize(avg, bitlevel);
	if (binary) fwrite(&avg, sizeof(float), 1, fo);
	else fprintf(fo, "%lf ", avg);
      }
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
  int i;
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-save-every-epoch", argc, argv)) > 0) save_every_epoch = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-bitlevel", argc, argv)) > 0) bitlevel = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-reg", argc, argv)) > 0) reg = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
