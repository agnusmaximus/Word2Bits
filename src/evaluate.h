#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <sstream>
#include <algorithm>
#include "word2bits.h"

using namespace std;

#define DATA_PATH "./data/google_analogies_test_set/questions-words.txt"

int evaluate_analogy(Vocabulary *v, const string &analogy, int debug_print) {
  stringstream in(analogy);
  string a, b, c, d;
  in >> a >> b >> c >> d;
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  transform(c.begin(), c.end(), c.begin(), ::tolower);
  transform(d.begin(), d.end(), d.begin(), ::tolower);

  if (v->word_to_index.find(a) == v->word_to_index.end()) {
    //cout << "Error finding a word in analogy: " << a << endl;
    return 0;
  }
  if (v->word_to_index.find(b) == v->word_to_index.end()) {
    //cout << "Error finding a word in analogy: " << b << endl;
    return 0;
  }
  if (v->word_to_index.find(c) == v->word_to_index.end()) {
    //cout << "Error finding a word in analogy: " << c << endl;
    return 0;
  }
  if (v->word_to_index.find(d) == v->word_to_index.end()) {
    //cout << "Error finding a word in analogy: " << d << endl;
    return 0;
  }  
  
  char *v_a = WordToBits(v, v->word_to_index[a], v->emb1);
  char *v_b = WordToBits(v, v->word_to_index[b], v->emb1);
  char *v_c = WordToBits(v, v->word_to_index[c], v->emb1);
  char *v_d = WordToBits(v, v->word_to_index[d], v->emb1);

  //char diff_v_a_v_b[BITSIZE / BITS_PER_BYTE];
  //char v_c_plus_diff[BITSIZE / BITS_PER_BYTE];
  //SUB(v_b, v_a, diff_v_a_v_b);
  //ADD(v_c, diff_v_a_v_b, v_c_plus_diff);
  //XOR(v_a, v_b, diff_v_a_v_b);
  //XOR(v_c, diff_v_a_v_b, v_c_plus_diff);
  char expected[BITSIZE/BITS_PER_BYTE];
  ANALOGY(v_a, v_b, v_c, v_d, expected);

  unsigned int min_score = ~0;
  int best_index = -1;
  for (int i = 0; i < v->n_unique_words; i++) {
    if (i == v->word_to_index[c]) continue;
    char *cur_bits = WordToBits(v, i, v->emb1);
    unsigned int score = (unsigned int)HammingDistance(cur_bits, expected);
    if (score < min_score) {
      min_score = score;
      best_index = i;
    }
  }

  // For debugging
  //if (debug_print) {
  if (analogy.find("England English Japan Japanese") != string::npos) {
    int true_word_distance_score = HammingDistance(expected, v_d);
    cout << "Expected:" << a << " : " << b << " | " << c << " : " << d << endl;
    cout << "Got     :" << a << " : " << b << " | " << c << " : " << v->index_to_word[best_index] << endl;
    printf("v_a v_b v_c v_d\n");
    PrintBits(v_a);
    PrintBits(v_b);
    PrintBits(v_c);
    PrintBits(v_d);
    cout << "Score: (expected)" << true_word_distance_score << " vs " << min_score << endl;
  }
  
  return best_index == v->word_to_index[d];
}

void evaluate_google_analogies(Vocabulary *v) {
  ifstream in(DATA_PATH);
  assert(in);
  map<string, vector<string>> tests;

  // Read data into tests dictionary
  string current_section = "";
  while (!in.eof()) {
    string inputline;
    getline(in, inputline);
    if (inputline[0] == ':') {
      cout << "Reading: " << inputline << endl;
      tests[inputline] = vector<string>();
      current_section = inputline;
    }
    else {
      tests[current_section].push_back(inputline);
    }
  }
  in.close();

  // Evaluate on test data
  map<string, vector<string>>::iterator it;
  map<string, pair<int, int>> score;
  for (map<string, vector<string>>::iterator it = tests.begin(); it != tests.end(); it++) {
    int score = 0, total = 0;
    for (unsigned int i = 0; i < it->second.size(); i++) {
      score += evaluate_analogy(v, it->second[i], i < 10);
      total++;
    }
    printf("%s: %d/%d\n", it->first.c_str(), score, total);
  }
}
