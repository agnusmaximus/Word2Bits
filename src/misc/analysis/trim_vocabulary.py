from __future__ import print_function
import sys
from convert_word2bits import *

print("Usage: trim_vocabulary.py input_word_vectors_path vocab_size corpus_path output_path")
assert len(sys.argv) == 5

input_path = sys.argv[1]
vocab_size = int(sys.argv[2])
corpus_path = sys.argv[3]
output_path = sys.argv[4]

word_vecs = load_vec(input_path)

word_counts = {}
print("Reading corpus...")
with open(corpus_path, "r") as f:
    for line in f:
        line = line.strip()
        for word in line.split(" "):
            word = word.strip()
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
print("Done reading corpus.")

# Sanity check
for i, (word,vec) in enumerate(word_vecs.items()):
    print("Sanity check %d of %d" % (i, len(word_vecs)))
    if word not in word_counts:
        print("Word not in word_counts:", word)

sorted_words = sorted(word_counts.keys(), key=lambda x:word_counts[x], reverse=True)
words_to_use = []
words_added = 0

for word in sorted_words:
    if words_added >= vocab_size:
        break
    print("%d written of %d" % (words_added, vocab_size))
    if word in word_vecs:
        words_to_use.append(word)
        words_added += 1
    else:
        print("Missing word: %s. Skipping" % word)

print("Writing out final words")
with open(output_path, "w") as f:
    print("%d %d" % (words_added, len(word_vecs.values()[0])), file=f)
    n_written = 0
    for word in words_to_use:
        print("%d written to disk of %d" % (n_written, words_added))
        n_written += 1
        vec = word_vecs[word]
        vec_string = " ".join([str(x) for x in vec])
        print("%s %s" % (word, vec_string), file=f)
    
