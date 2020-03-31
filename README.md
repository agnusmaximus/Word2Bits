# Word2Bits - Quantized Word Vectors

  Word2Bits extends the Word2Vec algorithm to output high quality
  quantized word vectors that take 8x-16x less storage than
  regular word vectors. Read the details at [https://arxiv.org/abs/1803.05651](https://arxiv.org/abs/1803.05651).

## What are Quantized Word Vectors?

  Quantized word vectors are word vectors where each parameter
  is one of `2^bitlevel` values.

  For example, the 1-bit quantized vector for "king" looks something
  like

  ```
  0.33333334 0.33333334 0.33333334 -0.33333334 -0.33333334 -0.33333334 0.33333334 0.33333334 -0.33333334 0.33333334 0.33333334 ...
  ```

  Since parameters are limited to one of `2^bitlevel` values, each parameter
  takes only `bitlevel` bits to represent; this drastically reduces
  the amount of storage that word vectors take.

## Download Pretrained Word Vectors

- All word vectors are in Glove/Fasttext format (format details [here](https://fasttext.cc/docs/en/english-vectors.html)). Files are compressed using gzip.

| # Bits per parameter        | Dimension     | Trained on             | Vocabulary size  | File Size (Compressed) | Download Link |
|:---------------------------:|:-------------:|:----------------------:|:----------------:|:----------------------:|:-------------:|
| 1                           | 800           | English Wikipedia 2017 | Top 400k         | 86M                    | [w2b_bitlevel1_size800_vocab400K.tar.gz](https://drive.google.com/open?id=107guTTy93J-y7UCO2ZA2spxRIFpoqhjh) |
| 1                           | 1000          | English Wikipedia 2017 | Top 400k         | 106M                   | [w2b_bitlevel1_size1000_vocab400K.tar.gz](https://drive.google.com/open?id=1boP7aFnABifVKRD9M-lxLqH-BM-C5-Z6) |
| 1                           | 1200          | English Wikipedia 2017 | Top 400k         | 126M                   | [w2b_bitlevel1_size1200_vocab400K.tar.gz](https://drive.google.com/open?id=1zmoHFd9KqsCvuvYqpMl0Si21wn9IHMq2) |
| 2                           | 400           | English Wikipedia 2017 | Top 400k         | 67M                    | [w2b_bitlevel2_size400_vocab400K.tar.gz](https://drive.google.com/open?id=1KHNDZW9dawwy9Ie73fdnMKAcGfadTI5J) |
| 2                           | 800           | English Wikipedia 2017 | Top 400k         | 134M                   | [w2b_bitlevel2_size800_vocab400K.tar.gz](https://drive.google.com/open?id=1l3G4tyI8mU7bGsMG0TTPiM4fucmniJaR) |
| 2                           | 1000          | English Wikipedia 2017 | Top 400k         | 168M                   | [w2b_bitlevel2_size1000_vocab400K.tar.gz](https://drive.google.com/open?id=1RX5z-jjpylAKTxpVazWqQmkDZ0XnumsB) |
| 32                          | 200           | English Wikipedia 2017 | Top 400k         | 364M                   | [w2b_bitlevel0_size200_vocab400K.tar.gz](https://drive.google.com/open?id=1HKiDirbJ9oxJN1HXGdczvmjWTIazE0Gb) |
| 32                          | 400           | English Wikipedia 2017 | Top 400k         | 724M                   | [w2b_bitlevel0_size400_vocab400K.tar.gz](https://drive.google.com/open?id=1ToIOpo0uhfGG48qsOZeDacPmt7Sh0Uup) |
| 32                          | 800           | English Wikipedia 2017 | Top 400k         | 1.4G                   | [w2b_bitlevel0_size800_vocab400K.tar.gz](https://drive.google.com/open?id=1IMev4MIQKSx5CPgGhTxZo2EJ7nsVEVGT) |
| 32                          | 1000          | English Wikipedia 2017 | Top 400k         | 1.8G                   | [w2b_bitlevel0_size1000_vocab400K.tar.gz](https://drive.google.com/open?id=1CtNjaQqK2Aw-iIeqRXdJOVNIqeALzOTi) |
| 1                           | 800           | English Wikipedia 2017 | 3.7M (Full)      | 812M                   | [w2b_bitlevel1_size800_vocab3.7M.tar.gz](https://drive.google.com/open?id=1fisO5pl3KbP5DEGqb3-b8RxOsbqzquZE) |
| 2                           | 400           | English Wikipedia 2017 | 3.7M (Full)      | 671M                   | [w2b_bitlevel2_size400_vocab3.7M.tar.gz](https://drive.google.com/open?id=139YwOxwhoIgKACXUJnfOdecxIueEkwf9) |
| 32                          | 400           | English Wikipedia 2017 | 3.7M (Full)      | 6.7G                   | [w2b_bitlevel0_size400_vocab3.7M.tar.gz](https://drive.google.com/open?id=1zyizh_oJ3RHtdaHdT_V7eQUTwETuXKm6) |

## Visualizing Quantized Word Vectors

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
./word2bits -train input -bitlevel 1 -size 200 -window 10 -negative 12 -threads 2 -iter 5 -min-count 5  -output 1bit_200d_vectors -binary 0
```
Description of the most common flags:
```
-train                       Input corpus text file
-bitlevel          	     Number of bits for each parameter. 0 is full precision (or 32 bits).
-size                        Word vector dimension
-window                      Window size
-negative                    Negative sample size
-threads                     Number of threads to use to train
-iter                        Number of epochs to train
-min-count                   Minimum count value. Words appearing less than value are removed from corpus.
-output                      Path to write output word vectors
-binary                      0 to write in Glove format; 1 to write in binary format.
```

### Example: Word2Bits on text8

1. Download and preprocess text8 (make sure you're in the Word2Bits base directory).
   ```
   bash data/download_text8.sh
   ```

2. Compile Word2Bits and compute accuracy
   ```
   make word2bits
   ```
   ```
   make compute_accuracy
   ```

3. Train 1 bit 200 dimensional word vectors for 5 epochs using 4 threads (save in binary so that compute_accuracy can work with it)
   ```
   ./word2bits -bitlevel 1 -size 200 -window 8 -negative 24 -threads 4 -iter 5 -min-count 5 -train text8  -output 1b200d_vectors -binary 1
   ```

   (This will take several minutes. Run with more threads if you have more cores!)

4. Evaluate vectors on Google Analogy Task
   ```
   ./compute_accuracy ./1b200d_vectors < data/google_analogies_test_set/questions-words.txt
   ```

   You should see output like:
   ```
   Starting eval...
   capital-common-countries:
   ACCURACY TOP1: 19.76 %  (100 / 506)
   Total accuracy: 19.76 %   Semantic accuracy: 19.76 %   Syntactic accuracy: -nan %
   capital-world:
   ACCURACY TOP1: 8.81 %  (239 / 2713)
   Total accuracy: 10.53 %   Semantic accuracy: 10.53 %   Syntactic accuracy: -nan %
   ...
   gram8-plural:
   ACCURACY TOP1: 19.92 %  (251 / 1260)
   Total accuracy: 11.48 %   Semantic accuracy: 13.27 %   Syntactic accuracy: 10.25 %
   gram9-plural-verbs:
   ACCURACY TOP1: 6.09 %  (53 / 870)
   Total accuracy: 11.20 %   Semantic accuracy: 13.27 %   Syntactic accuracy: 9.88 %
   Questions seen / total: 16284 19544   83.32 %
   ```

   Inspecting the vector file in hex should show something like:
   ```
   $ od --format=x1 --read-bytes=160 1b200d_vectors
   0000000 36 30 32 33 38 20 32 30 30 0a 3c 2f 73 3e 20 ab
   0000020 aa aa 3e ab aa aa 3e ab aa aa be ab aa aa be ab
   0000040 aa aa 3e ab aa aa 3e ab aa aa 3e ab aa aa be ab
   ...
   0000160 aa aa be ab aa aa 3e ab aa aa be ab aa aa 3e ab
   0000200 aa aa be ab aa aa 3e ab aa aa 3e ab aa aa 3e ab
   0000220 aa aa be ab aa aa 3e ab aa aa be ab aa aa 3e ab
   ```
