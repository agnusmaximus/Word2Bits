#!/bin/sh
mkdir word2vecf
wget https://bitbucket.org/yoavgo/word2vecf/get/1b94252a58d4.zip
unzip 1b94252a58d4.zip
rm 1b94252a58d4.zip
mv yoavgo-word2vecf-1b94252a58d4/*.c word2vecf/.
mv yoavgo-word2vecf-1b94252a58d4/*.h word2vecf/.
mv yoavgo-word2vecf-1b94252a58d4/makefile word2vecf/.
rm -r yoavgo-word2vecf-1b94252a58d4
make -C word2vecf
