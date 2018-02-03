#ifndef WORD2BITSH
#define WORD2BITSH

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

#define WINDOW_SIZE 8
#define NEGATIVE_WINDOW_SIZE 25
#define SIZE 200
#define LEARNING_RATE .05
#define SUBSAMPLING_COEFFICIENT 1e-4
#define N_WORKERS 1
#define N_EPOCHS_PER_WORKER 1e9
#define UNIGRAM_TABLE_SIZE 1e8
#define PRINT_INTERVAL 1000000
#define FAST_RAND_MAX 32768

using namespace std;

////////////////////////////////////////
//            MISC UTILS              //
////////////////////////////////////////

static unsigned int g_seed;

// Used to seed the generator.
inline void fast_srand(int seed) {
  g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int fast_rand(void) {
  g_seed = (214013*g_seed+2531011);
  return (g_seed>>16)&0x7FFF;
}

/////////////////////////////////////
//           Vocabulary            //
/////////////////////////////////////

// Vocabulary stores data about corpus and embeddings.
struct Vocabulary {
  unsigned long n_total_words;
  int n_unique_words;
  unordered_map<string, int> word_to_index;
  unordered_map<int, string> index_to_word;
  unordered_map<int, int> word_index_to_count;
  unsigned long filesize;
  vector<int> unigram_table;

  // Embeddings
  double *emb;

  // Embeddings weights
  double *weights;
};
typedef struct Vocabulary Vocabulary;

double * WordEmbeddings(Vocabulary *vocab, int word_index) {
  return &vocab->emb[word_index*SIZE];
}

void Update(Vocabulary *v, int word_index) {
  memcpy(&v->emb[word_index*SIZE],
	 &v->weights[word_index*SIZE],
	 sizeof(double) * SIZE);
}

// Build vocabulary from text file.
Vocabulary *CreateVocabulary(const char *filepath) {

  printf("Creating vocabulary...\n");
  time_t tstart = time(0);

  // Create and initialize vocab
  Vocabulary *vocab = new Vocabulary();
  vocab->n_unique_words = vocab->n_total_words = 0;
  vocab->word_to_index.reserve(100000);
  vocab->index_to_word.reserve(100000);
  vocab->emb = NULL;
  vocab->filesize = 0;

  // Create hashmaps from words to indices
  ifstream file(filepath);
  while (!file.eof()) {
    string cur_word;
    file >> cur_word;

    unordered_map<string, int>::iterator it = vocab->word_to_index.find(cur_word);
    if (it == vocab->word_to_index.end()) {
      vocab->word_to_index[cur_word] = vocab->n_unique_words;
      vocab->index_to_word[vocab->n_unique_words] = cur_word;
      vocab->word_index_to_count[vocab->n_unique_words] = 1;
      vocab->n_unique_words++;
    }
    else {
      vocab->word_index_to_count[it->second]++;
    }
    vocab->n_total_words++;
    vocab->filesize += cur_word.size() + 1;
  }

  // Allocate and randomize embeddings
  vocab->emb = (double *)malloc(sizeof(double) * (vocab->n_unique_words * SIZE));
  vocab->weights = (double *)malloc(sizeof(double) * (vocab->n_unique_words * SIZE));;
  for (int i = 0; i < vocab->n_unique_words; i++) {
    for (int j = 0; j < SIZE; j++) {
      vocab->emb[i*SIZE+j] = (((double)fast_rand() / (FAST_RAND_MAX)) - .5)/SIZE;
      vocab->weights[i*SIZE+j] = 0;
    }
    //Update(vocab, i);
  }

  // Generate negative sampling unigram table.
  vocab->unigram_table.reserve(UNIGRAM_TABLE_SIZE);
  for (int i = 0; i < vocab->n_unique_words; i++) {
    double weight = vocab->word_index_to_count[i] / (double)vocab->n_total_words;
    int n_occ = weight * UNIGRAM_TABLE_SIZE;
    for (int j = 0; j < n_occ; j++) {
      vocab->unigram_table.push_back(i);
    }
  }
  printf("YOO: %d\n", vocab->unigram_table.size());

  printf("- Vocabulary size : %d\n", vocab->n_unique_words);
  printf("- File size : %ld bytes\n", vocab->filesize);

  file.close();

  time_t tend = time(0);
  printf("Done creating vocabulary. (%lfs)\n", difftime(tend, tstart));

  return vocab;
}

int Keep(Vocabulary *v, int word) {
  if (word < 0) return 0;
  double z_w = v->word_index_to_count[word] / (double)v->n_total_words;
  double p_w = (sqrt(z_w/(double)SUBSAMPLING_COEFFICIENT) + 1) * (SUBSAMPLING_COEFFICIENT / z_w);
  return ((double)fast_rand() / (FAST_RAND_MAX)) <= p_w;
}

int NegativeSample(Vocabulary *v) {
  return v->unigram_table[fast_rand() % v->unigram_table.size()];
}

int WordToIndex(Vocabulary *vocab, string &word) {
  return vocab->word_to_index[word];
}

// Destroy vocabulary
void DestroyVocabulary(Vocabulary *vocab) {
  free(vocab->weights);
  free(vocab->emb);
  delete vocab;
}

/////////////////////////////////////////////////
//                 Context                     //
/////////////////////////////////////////////////

struct Context {
  int size;
  int counter;
  double **context;
  int *word_ids;
  int did_wrap;
  Vocabulary *vocab;
  double counts[SIZE];
};
typedef struct Context Context;

Context * CreateContext(Vocabulary *v, int size) {
  Context *context = new Context();
  context->size = size;
  context->vocab = v;
  context->counter = 0;
  context->did_wrap = 0;
  context->context = (double **)malloc(sizeof(double *) * size);
  context->word_ids = (int *)malloc(sizeof(int) * size);
  memset(context->context, 0, sizeof(double *) * size);
  memset(context->counts, 0, sizeof(double) * SIZE);
  memset(context->word_ids, 0, sizeof(int) * size);
  for (int i = 0; i < size; i++) {
    context->context[i] = (double *)malloc(sizeof(double) * SIZE);
  }
  return context;
}

int CurrentWordOfContext(Context *c) {
  return c->word_ids[c->counter];
}

int CenterWordOfContext(Context *c) {
  return c->word_ids[(c->counter + c->size/2) % c->size];
}

void AddWordToContext(Context *c, int word_id, double *embedding) {

  for (int i = 0; i < SIZE; i++) {
    c->counts[i] += embedding[i] - c->did_wrap * c->context[c->counter][i];
  }

  // Update counters
  c->word_ids[c->counter] = word_id;
  memcpy(c->context[c->counter++],
	 embedding,
	 sizeof(double) * SIZE);
  c->did_wrap = c->did_wrap | (c->counter==c->size);
  c->counter %= c->size;
}

void DestroyContext(Context *c) {
  for (int i = 0; i < c->size; i++)
    free(c->context[i]);
  free(c->context);
  free(c->word_ids);
  delete c;
}

#endif
