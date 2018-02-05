#!/bin/bash

fname="enwiki-latest-pages-articles.xml.bz2"
source="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

if [ -e ${fname} ]
then
    echo "Found wikipedia dump, continuing..."
else
    echo "Downloading wikipedia xml..."
    wget ${source}
fi

python process_wiki.py enwiki-latest-pages-articles.xml.bz2 wiki.en.text
