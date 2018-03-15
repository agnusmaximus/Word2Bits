wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip
perl data/wikifil.pl enwik8 > text8
rm -f enwik8.zip
rm -f enwik8
