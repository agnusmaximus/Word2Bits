import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import sys
import matplotlib.pyplot as plt
from convert_word2bits import *

fname = sys.argv[1]
word_vecs = load_vec(fname)

def visualize_vector(word):
    assert word in word_vecs
    dimension = len(word_vecs[word])
    dimension = int(math.ceil(math.sqrt(dimension)))
    remaining = dimension**2 - len(word_vecs[word])
    vector = np.hstack([word_vecs[word], np.zeros(remaining)])
    vector = vector.reshape((dimension, dimension))
    plt.cla()
    plt.imshow(vector)
    plt.savefig("visualize_%s.png" % (word))

def visualize_diff(word1, word2):
    assert word1 in word_vecs
    assert word2 in word_vecs
    dimension = len(word_vecs[word1])
    dimension = int(math.ceil(math.sqrt(dimension)))
    remaining = dimension**2 - len(word_vecs[word1])
    v1, v2 = np.array(word_vecs[word1]), np.array(word_vecs[word2])
    vector = v1 - v2
    vector = np.concatenate([vector, np.zeros(remaining)])
    vector = vector.reshape((dimension, dimension))
    plt.cla()
    plt.imshow(vector)
    plt.savefig("visualize_diff_%s_%s" % (word1, word2))

def visualize_nearest(seed, nn=100):
    stacked_vectors = np.stack(word_vecs.values())
    seedv = np.array(word_vecs[seed])
    correlation = stacked_vectors.dot(seedv)

    sorted_words = sorted(enumerate(word_vecs.items()), key=lambda x: correlation[x[0]], reverse=True)
    stacked_vectors = np.stack([x[1][1] for x in sorted_words[:nn]] + [[0 for i in range(len(sorted_words[0][1][1]))]] + [x[1][1] for x in sorted_words[-nn:]])
    print([x[1][0] for x in sorted_words[:nn]])
    print([x[1][0] for x in sorted_words[-nn:]])
    labels = [x[1][0] for x in sorted_words[:nn]] + ["..."] + [x[1][0] for x in sorted_words[-nn:]]
    plt.cla()
    rcParams['figure.figsize'] = 5, 10
    label_ticks = list(range(0, nn*2, 5))
    labels = [labels[i] for i in label_ticks]
    plt.yticks(label_ticks, labels, fontsize=8)
    plt.title("Nearest and furthest neighbors of \"%s\"" % seed)
    plt.xlabel("Word Vector Dimension")
    plt.imshow(stacked_vectors, aspect="auto")
    plt.tight_layout()
    plt.savefig("visualize_nearest_%s.pdf" % seed)
    
visualize_nearest("man")
visualize_nearest("science")
visualize_nearest("mushroom")
visualize_nearest("language")
            
