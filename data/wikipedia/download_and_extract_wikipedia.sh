
source="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

wget ${source}

python process_wiki.py enwiki-latest-pages-articles.xml.bz2 wiki.en.text
