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

#define WINDOW_SIZE 10
#define NEGATIVE_WINDOW_SIZE 100
#define BITSIZE 128
#define LEARNING_RATE .05
#define SUBSAMPLING_COEFFICIENT 1e-3
#define BITS_PER_BYTE 8
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

unsigned long long SwapLong(unsigned long long X) {
  uint64_t x = (uint64_t) X;
  x = (x & 0x00000000FFFFFFFF) << 32 | (x & 0xFFFFFFFF00000000) >> 32;
  x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16;
  x = (x & 0x00FF00FF00FF00FF) << 8  | (x & 0xFF00FF00FF00FF00) >> 8;
  return x;
}

void SetBitAtIndex(char *embedding, int index, int value) {
    int char_index = index / BITS_PER_BYTE;
    char mask = (char)1 << (BITS_PER_BYTE-1-index%BITS_PER_BYTE);
    embedding[char_index] &= ~mask;
    embedding[char_index] |= value * mask;
}

int ExtractBitAtIndex(char *embedding, int index) {
    int char_index = index / BITS_PER_BYTE;
    return (embedding[char_index] >> (BITS_PER_BYTE-1-index%BITS_PER_BYTE)) & 1;
}

void PrintBits(char *embedding) {
    for (int i = 0; i < BITSIZE; i++) {
	printf("%d", ExtractBitAtIndex(embedding, i));
    }
    printf("\n");
}

void XOR(char *v1, char *v2, char *out) {
  for (int i = 0; i < BITSIZE/BITS_PER_BYTE; i++) 
    out[i] = v1[i] ^ v2[i];
}

void SUB(char *v1, char *v2, char *out) {
  for (int i = 0; i < BITSIZE; i++) {
    int b1 = ExtractBitAtIndex(v1, i);
    int b2 = ExtractBitAtIndex(v2, i);
    int res = b1 - b2;
    SetBitAtIndex(out, i, res < 0 ? 0 : res);
  }
}

void ANALOGY(char *v1, char *v2, char *v3, char *v4, char *out) {
  // v1:v2, v3:v4
  for (int i = 0; i < BITSIZE; i++) {
    int a1 = ExtractBitAtIndex(v1, i);
    int a2 = ExtractBitAtIndex(v2, i);
    int a3 = ExtractBitAtIndex(v3, i);
    int diff = a2 - a1;
    int expected = min(1, max(0, a3 + diff));
    SetBitAtIndex(out, i, expected);
  }
}

void ADD(char *v1, char *v2, char *out) {
  for (int i = 0; i < BITSIZE; i++) {
    int b1 = ExtractBitAtIndex(v1, i);
    int b2 = ExtractBitAtIndex(v2, i);
    int res = b1 + b2;
    SetBitAtIndex(out, i, res > 1 ? 1 : res);
  }
}

int HammingDistance(char *v1, char *v2) {
  int s_diff = 0;
  for (int i = 0; i < BITSIZE/BITS_PER_BYTE; i++) 
    s_diff += __builtin_popcount((unsigned char)(v1[i] ^ v2[i]));
  return s_diff;
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
    char *emb1;

    // Embeddings weights
    float *weights;
};
typedef struct Vocabulary Vocabulary;

char * WordToBits(Vocabulary *vocab, int word_index, char *emb) {
    return &emb[word_index*BITSIZE/BITS_PER_BYTE];
}

void UpdateBits(Vocabulary *v, int word_index) {
  for (int i = 0; i < BITSIZE; i += BITS_PER_BYTE) {
    char to_set = 0;
    for (int j = 0; j < BITS_PER_BYTE; j++) {
      to_set <<= 1;
      to_set |= (char)(v->weights[word_index*BITSIZE+i+j] > 0);
    }
    v->emb1[word_index*BITSIZE/BITS_PER_BYTE + i/BITS_PER_BYTE] = to_set;
  }
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
    vocab->emb1 = NULL;
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
    int nchars = vocab->n_unique_words*BITSIZE/BITS_PER_BYTE;
    vocab->emb1 = (char *)malloc(sizeof(char) * nchars);
    vocab->weights = (float *)malloc(sizeof(float) * (vocab->n_unique_words * BITSIZE));;
    for (int i = 0; i < vocab->n_unique_words; i++) {
      for (int j = 0; j < BITSIZE; j++) {
	vocab->weights[i*BITSIZE+j] = ((float)fast_rand() / (FAST_RAND_MAX)) - .5;
	SetBitAtIndex(WordToBits(vocab, i, vocab->emb1), j, vocab->weights[i*BITSIZE+j] > 0);
      }
    }

    // Generate negative sampling unigram table.
    vocab->unigram_table.reserve(UNIGRAM_TABLE_SIZE);
    for (int i = 0; i < vocab->n_unique_words; i++) {
	float weight = vocab->word_index_to_count[i] / (float)vocab->n_total_words;
	int n_occ = weight * UNIGRAM_TABLE_SIZE;
	for (int j = 0; j < n_occ; j++) {
	    vocab->unigram_table.push_back(i);
	}
    }

    printf("- Vocabulary size : %d\n", vocab->n_unique_words);
    printf("- File size : %ld bytes\n", vocab->filesize);

    file.close();

    time_t tend = time(0);
    printf("Done creating vocabulary. (%lfs)\n", difftime(tend, tstart));

    return vocab;
}

int Keep(Vocabulary *v, int word) {
    if (word < 0) return 0;
    float z_w = v->word_index_to_count[word] / (float)v->n_total_words;
    float p_w = (sqrt(z_w/(float)SUBSAMPLING_COEFFICIENT) + 1) * (SUBSAMPLING_COEFFICIENT / z_w);
    return ((float)fast_rand() / (FAST_RAND_MAX)) <= p_w;
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
    free(vocab->emb1);
    delete vocab;
}

/////////////////////////////////////////////////
//                 Context                     //
/////////////////////////////////////////////////

// A circular buffer containing words and their bitcounts.
struct Context {
    int size;
    int counter;
    char **context;
    int *word_ids;
    int did_wrap;
    Vocabulary *vocab;
    char bitcounts[BITSIZE];
};
typedef struct Context Context;

Context * CreateContext(Vocabulary *v, int size) {
    Context *context = new Context();
    context->size = size;
    context->vocab = v;
    context->counter = 0;
    context->did_wrap = 0;
    context->context = (char **)malloc(sizeof(char *) * size);
    context->word_ids = (int *)malloc(sizeof(int) * size);
    memset(context->context, 0, sizeof(char *) * size);
    memset(context->bitcounts, 0, sizeof(char) * BITSIZE);
    memset(context->word_ids, 0, sizeof(int) * size);
    for (int i = 0; i < size; i++) {
      context->context[i] = (char *)malloc(sizeof(char) * BITSIZE/BITS_PER_BYTE);
    }
    return context;
}

int CurrentWordOfContext(Context *c) {
    return c->word_ids[c->counter];
}

int CenterWordOfContext(Context *c) {
    return c->word_ids[(c->counter + c->size/2) % c->size];
}

char * CurrentWordBits(Context *c) {
    return c->context[c->counter];
}

void AddWordToContext(Context *c, int word_id, char *embedding) {


#ifdef BMI_ENABLED
    unsigned long long *bitcounts_index = (unsigned long long *)c->bitcounts;
    unsigned long long mask = 0x0101010101010101ULL;    
    for (int i = 0; i < BITSIZE; i += BITS_PER_BYTE) {      
      unsigned long long to_add = SwapLong(_pdep_u64((unsigned long long)embedding[i/BITS_PER_BYTE],
						     mask));
      unsigned long long to_sub = SwapLong(_pdep_u64((unsigned long long )c->context[c->counter][i/BITS_PER_BYTE],
						     mask));
      *bitcounts_index = *bitcounts_index - to_sub * (unsigned long long )c->did_wrap + to_add;      
      bitcounts_index++;
    }
#else        
    for (int i = 0; i < BITSIZE; i++) {
      c->bitcounts[i] += ExtractBitAtIndex(embedding, i) - 
      (ExtractBitAtIndex(c->context[c->counter], i) & c->did_wrap);
    }
#endif    
    
    // Update counters
    c->word_ids[c->counter] = word_id;
    memcpy(c->context[c->counter++],
	   embedding,
	   sizeof(char) * BITSIZE / BITS_PER_BYTE);
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

// More misc utils
void PrintVectorDifference(Vocabulary *v, int first, int second) {
  char *b1 = WordToBits(v, first, v->emb1);
  char *b2 = WordToBits(v, second, v->emb1);
  char diff[BITSIZE];
  int s_diff = 0;
  for (int i = 0; i < BITSIZE/BITS_PER_BYTE; i++) {
    diff[i] = b1[i] ^ b2[i];
    int popcount = __builtin_popcount((unsigned char)diff[i]);
    assert(popcount <= 8);
    s_diff += popcount;
  }
  PrintBits(diff);
  printf("diff: %d\n", s_diff);
}

#endif
