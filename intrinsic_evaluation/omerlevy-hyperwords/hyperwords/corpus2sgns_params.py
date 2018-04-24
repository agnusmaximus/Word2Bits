from docopt import docopt


# noinspection PyListCreation
def main():
    args = docopt("""
    Usage:
        corpus2sgns.sh [options] <corpus> <output_dir>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
        --pos        Positional contexts
        --dyn        Dynamic context windows
        --sub NUM    Subsampling threshold [default: 0]
        --del        Delete out-of-vocabulary and subsampled placeholders
        --cds NUM    Context distribution smoothing [default: 1.0]
        --dim NUM    Dimensionality of embeddings [default: 500]
        --neg NUM    Number of negative samples; subtracts its log from PMI [default: 1]
        --w+c        Use ensemble of word and context vectors
        --cpu NUM    The number of threads when training SGNS [default: 1]
    """)
    
    corpus = args['<corpus>']
    output_dir = args['<output_dir>']

    corpus2pairs_opts = []
    corpus2pairs_opts.append('--thr ' + args['--thr'])
    corpus2pairs_opts.append('--win ' + args['--win'])
    if args['--pos']:
        corpus2pairs_opts.append('--pos')
    if args['--dyn']:
        corpus2pairs_opts.append('--dyn')
    corpus2pairs_opts.append('--sub ' + args['--sub'])
    if args['--del']:
        corpus2pairs_opts.append('--del')

    word2vecf_opts = []
    word2vecf_opts.append('-pow ' + args['--cds'])
    word2vecf_opts.append('-size ' + args['--dim'])
    word2vecf_opts.append('-negative ' + args['--neg'])
    word2vecf_opts.append('-threads ' + args['--cpu'])

    sgns2text_opts = []
    if args['--w+c']:
        sgns2text_opts.append('--w+c')

    print '@'.join([
        corpus,
        output_dir,
        ' '.join(corpus2pairs_opts),
        ' '.join(word2vecf_opts),
        ' '.join(sgns2text_opts)
    ])


if __name__ == '__main__':
    main()
