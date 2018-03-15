# Word2Bits - Quantized Word Vectors

  Word2Bits extends the Word2Vec algorithm to output high quality
  quantized word vectors that take 8x-16x less storage/memory than
  regular word vectors.

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

## Visualizing Quantized Word Vectors?

Here are examples of 1 bit word vectors and their nearest and furthest neighbors:

![Word Vector for "man"](images/visualize_nearest_man.png?raw=true =250x) ![Word Vector for "science"](images/visualize_nearest_science.png?raw=true =250x)


## Using the Code

### Quickstart

### Tutorial - text8