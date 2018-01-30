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

#define WINDOW_SIZE 16
#define NEGATIVE_WINDOW_SIZE 8
#define BITSIZE 256
#define SUBSAMPLING_COEFFICIENT 1e-3
#define BIT_LEARNING_RATE .6
#define BITS_PER_BYTE 8
#define N_WORKERS 1
#define N_EPOCHS_PER_WORKER 1e9
#define UNIGRAM_TABLE_SIZE 1e8

using namespace std;

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
    char *emb1, *emb2;
};
typedef struct Vocabulary Vocabulary;

// Build vocabulary from text file.
Vocabulary *CreateVocabulary(const char *filepath) {

    printf("Creating vocabulary...\n");
    time_t tstart = time(0);

    // Create and initialize vocab
    Vocabulary *vocab = new Vocabulary();
    vocab->n_unique_words = vocab->n_total_words = 0;
    vocab->word_to_index.reserve(100000);
    vocab->index_to_word.reserve(100000);
    vocab->emb1 = vocab->emb2 = NULL;
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
    vocab->emb2 = new char[nchars];
    for (int i = 0; i < nchars/sizeof(int); i++) {
	vocab->emb1[i*sizeof(int)] = rand();
	vocab->emb2[i*sizeof(int)] = rand();
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

char * WordToBits(Vocabulary *vocab, int word_index, char *emb) {
    return &emb[word_index*BITSIZE/BITS_PER_BYTE];
}

// Destroy vocabulary
void DestroyVocabulary(Vocabulary *vocab) {
    delete vocab->emb1;
    delete vocab->emb2;
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
    context->context = new char *[size];
    context->word_ids = new int[size];
    memset(context->context, 0, sizeof(char *) * size);
    memset(context->bitcounts, 0, sizeof(char) * BITSIZE);
    memset(context->word_ids, 0, sizeof(int) * size);
    return context;
}

void SetBitAtIndex(char *embedding, int index, int value) {
    int char_index = index / BITS_PER_BYTE;
    char mask = 1 << (BITS_PER_BYTE-1-index);
    embedding[char_index] &= ~mask;
    embedding[char_index] |= value * mask;
}

int ExtractBitAtIndex(char *embedding, int index) {
    int char_index = index / BITS_PER_BYTE;
    char mask = 1 << (BITS_PER_BYTE-1-index);
    return (embedding[char_index] & mask) != 0;
}

void ExtractBitcountsFromEmbedding(char *embedding, char *output) {
    for (int i = 0; i < BITSIZE; i++) {
	output[i] = ExtractBitAtIndex(embedding, i);
    }
}

// output -= input
void EmbeddingSubtract(char *output, char *input) {
    for (int i = 0; i < BITSIZE; i++) {
	output[i] -= input[i];
    }
}

// output += input
void EmbeddingAdd(char *output, char *input) {
    for (int i = 0; i < BITSIZE; i++) {
	output[i] += input[i];
    }
}

int CenterWordOfContext(Context *c) {
    return c->word_ids[(c->counter + c->size/2) % c->size];
}

void AddWordToContext(Context *c, int word_id, char *embedding) {
    char embedding_bitcounts[BITSIZE];

    if (c->did_wrap) {
	// Subtract current word from bitcounts
	ExtractBitcountsFromEmbedding(c->context[c->counter], embedding_bitcounts);
	EmbeddingSubtract(c->bitcounts, embedding_bitcounts);
    }

    // Add new word to bitcounts
    ExtractBitcountsFromEmbedding(embedding, embedding_bitcounts);
    EmbeddingAdd(c->bitcounts, embedding_bitcounts);

    // Update counters
    c->word_ids[c->counter] = word_id;
    c->context[c->counter++] = embedding;
    c->did_wrap = c->did_wrap | (c->counter==c->size);
    c->counter %= c->size;
}

void DestroyContext(Context *c) {
    delete c->context;
    delete c;
}

////////////////////////////////////////

// Worker train function
void TrainWorker(const char *filepath, int id, Vocabulary *vocab) {

    printf("Worker %d beginning training...\n", id);

    // Workers process file in parallel (but in different starting positions)
    unsigned long seek_pos = (id / N_WORKERS) * vocab->filesize;
    ifstream fin(filepath);
    fin.seekg(seek_pos);

    // Embeddings (always read from emb1, write to emb2. Swap periodically.)
    char *emb1, *emb2;
    emb1 = vocab->emb1;
    emb2 = vocab->emb2;

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

	// Update center word in positive context
	int center_word_index = CenterWordOfContext(positive_context);
	char *to_update = WordToBits(vocab, center_word_index, emb2);
	for (int i = 0; i < BITSIZE; i++) {

	    // Calculate counts of 1s for this bit
	    int self_count = ExtractBitAtIndex(emb1, i);
	    int positive_counts = positive_context->bitcounts[i];
	    int negative_counts = negative_context->bitcounts[i];
	    int weight = positive_counts + (NEGATIVE_WINDOW_SIZE-negative_counts) - self_count;

	    // Calculate loss
	    double neg_loss = self_count * weight + (1-self_count) * (WINDOW_SIZE + NEGATIVE_WINDOW_SIZE - weight);
	    negative_running_loss += neg_loss / (double)(WINDOW_SIZE + NEGATIVE_WINDOW_SIZE);

	    // Maximum weight is WINDOW_SIZE + NEGATIVE_WINDOW_SIZE (all of them are the same value)
	    double norm_weight = weight / (double)(WINDOW_SIZE + NEGATIVE_WINDOW_SIZE);
	    if (norm_weight > BIT_LEARNING_RATE) {
		// Set to 1
		SetBitAtIndex(emb2, i, 1);

	    }
	    else if (norm_weight < 1-BIT_LEARNING_RATE) {
		// Set to 0
		SetBitAtIndex(emb2, i, 0);
	    }
	}

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

	if (words_processed % 100000 == 0) {
	    time_t tcur = time(0);
	    double elapsed = difftime(tcur, tstart);
	    if (elapsed != 0) {
		printf("- %f words/sec (epoch %d) (neg loss %lf)\n",
		       words_processed/elapsed, n_epochs_processed+1,
		       negative_running_loss / 100000 / BITSIZE);
	    }
	    negative_running_loss = 0;
	}

	// Swap emb1 and emb2
	char *tmp = emb1;
	emb1 = emb2;
	emb2 = tmp;
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
