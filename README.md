# Word2Bits - Quantized Word Vectors

  Word2Bits extends the Word2Vec algorithm to output high quality
  quantized word vectors that take 8x-16x less storage/memory than
  regular word vectors. Read the details at [https://arxiv.org/abs/1803.05651](https://arxiv.org/abs/1803.05651).

## What are Quantized Word Vectors?

  Quantized word vectors are word vectors where each parameter
  is one of `2^bitlevel` values.

  For example, the 1-bit quantized vector for "king" looks something
  like

  ```
  0.33333334 0.33333334 0.33333334 -0.33333334 -0.33333334 -0.33333334 0.33333334 0.33333334 -0.33333334 0.33333334 0.33333334 ...
  ```

  Since parameters are limited to 1 of `2^bitlevel` values, each one
  takes only `bitlevel` bits to represent; this drastically reduces
  the amount of storage that word vectors take.

## Download Pretrained Word Vectors
  TODO

## Visualizing Quantized Word Vectors

Here are examples of 800 dimensional 1 bit word vectors and their nearest and furthest neighbors:

<img src="images/visualize_nearest_man.png?raw=true" width="400" height="300"/> <img src="images/visualize_nearest_science.png?raw=true" width="400" height="300"/>

(Note: every 5 word vectors are labelled; turquoise line boundary between nearest and furthest word vectors from target.)

## Using the Code

### Quickstart

Compile with
```
make word2bits
```

Run with
```
./word2bits -bitlevel 1 -size 200 -window 10 -negative 12 -threads 2 -iter 5 -min-count 5 -train inupt  -output 1bit_800d_vectors -binary 0
```
Description of the most common flags
```
- quantization_level - Number of bits for each parameter. 0 is full precision (or 32 bits).
- size - word vector dimension
- window - window size
- negative - negative sample size
- threads - number of threads to use to train
- iter - number of epochs to train
- min-count - min count. Words that appear less than min-count times in the corpus will be removed from the vocabulary.
- train - input corpus text file
- output - path to write output word vectors
- binary - whether to write output in binary or in text (Glove) format. 1 means write in binary, 0 means write in Glove format.
```

### Tutorial - text8