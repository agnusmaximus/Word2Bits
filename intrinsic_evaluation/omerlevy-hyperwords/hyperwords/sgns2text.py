from docopt import docopt

from representations.embedding import Embedding, EnsembleEmbedding


def main():
    args = docopt("""
    Usage:
        sgns2text.py [options] <sgns_path> <output_path>
    
    Options:
        --w+c        Use ensemble of word and context vectors
    """)
    
    sgns_path = args['<sgns_path>']
    output_path = args['<output_path>']
    w_c = args['--w+c']
    
    if w_c:
        sgns = EnsembleEmbedding(Embedding(sgns_path + '.words', False), Embedding(sgns_path + '.contexts', False), True)
    else:
        sgns = Embedding(sgns_path + '.words', True)
    
    with open(output_path, 'w') as f:
        for i, w in enumerate(sgns.iw):
            print >>f, w, ' '.join([str(x) for x in sgns.m[i]])


if __name__ == '__main__':
    main()
