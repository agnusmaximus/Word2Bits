#define NDEBUG

#include <iostream>
#include <math.h>
#include <vector>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <map>
#include <ctime>
#include <time.h>

#define WINDOW_SIZE 32
#define NEGATIVE_WINDOW_SIZE 32
#define BITSIZE 128
#define SUBSAMPLING_COEFFICIENT 1e-3
#define BITS_PER_BYTE 8
#define N_WORKERS 1
#define N_EPOCHS_PER_WORKER 1e9
#define UNIGRAM_TABLE_SIZE 1e8

using namespace std;


////////////////////////////////////////
//            MISC UTILS              //
////////////////////////////////////////
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

/////////////////////////////////////
//           Vocabulary            //
/////////////////////////////////////
struct Vocabulary {
    unsigned long n_unique_words, n_total_words;
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
      to_set |= (char)(v->weights[word_index*BITSIZE+i+j] >= .5);
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
    vocab->emb1 = new char[nchars];
    vocab->weights = new float[vocab->n_unique_words * BITSIZE];
    for (int i = 0; i < vocab->n_unique_words; i++) {
      for (int j = 0; j < BITSIZE; j++) {
	vocab->weights[i*BITSIZE+j] = ((double) rand() / (RAND_MAX));
	SetBitAtIndex(WordToBits(vocab, i, vocab->emb1), j, vocab->weights[i*BITSIZE+j] >= .5);
      }
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

    printf("- Vocabulary size : %ld\n", vocab->n_unique_words);
    printf("- File size : %ld bytes\n", vocab->filesize);

    file.close();

    time_t tend = time(0);
    printf("Done creating vocabulary. (%lfs)\n", difftime(tend, tstart));

    return vocab;
}

int Keep(Vocabulary *v, int word) {
    if (word < 0) return 0;
    double z_w = v->word_index_to_count[word] / (double)v->n_total_words;
    double p_w = (sqrt(z_w/(double).001) + 1) * (.001 / z_w);
    return ((double) rand() / (RAND_MAX)) <= p_w;
}

int NegativeSample(Vocabulary *v) {
    return v->unigram_table[rand() % v->unigram_table.size()];
}

int WordToIndex(Vocabulary *vocab, string &word) {
    return vocab->word_to_index[word];
}

// Destroy vocabulary
void DestroyVocabulary(Vocabulary *vocab) {
    delete vocab->weights;
    delete vocab->emb1;
    delete vocab;
}

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
    context->context = new char *[size];
    context->word_ids = new int[size];
    memset(context->context, 0, sizeof(char *) * size);
    memset(context->bitcounts, 0, sizeof(char) * BITSIZE);
    memset(context->word_ids, 0, sizeof(int) * size);
    for (int i = 0; i < size; i++) {
	context->context[i] = new char[BITSIZE/BITS_PER_BYTE];
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
    char embedding_bitcounts[BITSIZE];
    for (int i = 0; i < BITSIZE; i++) {
      embedding_bitcounts[i] = ExtractBitAtIndex(embedding, i);
      c->bitcounts[i] += embedding_bitcounts[i] -
	ExtractBitAtIndex(c->context[c->counter], i) * c->did_wrap;
    }
  
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
	delete c->context[i];
    delete c->context;
    delete c->word_ids;
    delete c;
}


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
    double negative_running_loss = 0;
    time_t tstart = time(0);
    while (n_epochs_processed != N_EPOCHS_PER_WORKER) {
      	if (words_processed % 500000 == 0) {
	    time_t tcur = time(0);
	    double elapsed = difftime(tcur, tstart);
	    if (elapsed >= 0) {
		printf("- %f words/sec (epoch %d) (neg loss %lf)\n",
		       words_processed/elapsed, n_epochs_processed+1,
		       negative_running_loss / 500000 / BITSIZE);
	    }
	    
	    negative_running_loss = 0;

	    int i1 = vocab->word_to_index["one"];
	    int i2 = vocab->word_to_index["two"];
	    int i3 = vocab->word_to_index["dog"];
	    int i4 = vocab->word_to_index["three"];
	    PrintVectorDifference(vocab, i1, i2);
	    PrintVectorDifference(vocab, i1, i3);
	    PrintVectorDifference(vocab, i1, i4);
	    PrintVectorDifference(vocab, i3, i4);
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

	// Update center word in positive context
	int center_word_index = CenterWordOfContext(positive_context);
	for (int i = 0; i < BITSIZE; i += BITS_PER_BYTE) {
	  char cur_byte = CurrentWordBits(positive_context)[i/BITS_PER_BYTE];
	  unsigned char mask = 1 << (BITS_PER_BYTE-1);
	  for (int k = 0; k < BITS_PER_BYTE; k++) {
	    
	    // Calculate counts of 1s for this bit
	    int self_count = (cur_byte & mask) != 0;
	    mask >>= 1;
	    int positive_counts = positive_context->bitcounts[i+k];
	    int negative_counts = negative_context->bitcounts[i+k];
	    double weight = positive_counts/(double)WINDOW_SIZE - self_count/(double)WINDOW_SIZE
	      + (NEGATIVE_WINDOW_SIZE-negative_counts)/(double)NEGATIVE_WINDOW_SIZE;
	    
	    // Calculate loss
	    double neg_loss =
	      ((self_count * positive_counts) +
	       (1-self_count) * (WINDOW_SIZE-positive_counts)) / (double)WINDOW_SIZE +
	      ((self_count * (NEGATIVE_WINDOW_SIZE - negative_counts)) +
	       (1-self_count) * (negative_counts)) / (double)NEGATIVE_WINDOW_SIZE;
	    negative_running_loss += neg_loss;

	    // Divide weight by 2 (half of the gradient comes from positive context, half from negative)
	    double norm_weight = weight / 2;
	    double grad = norm_weight - .5;
	    vocab->weights[center_word_index*BITSIZE + i+k] += .1 * grad;

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
	int to_write_word_index = CurrentWordOfContext(positive_context);
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

	// Reset file pointer if read to end.
	if (fin.eof()) {
	  fin.close();
	  fin.open(filepath);
	  n_epochs_processed++;
	}

	words_processed++;
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
