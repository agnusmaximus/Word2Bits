from __future__ import print_function
import sys
import numpy as np

def load_bin_vec(fname):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print(vocab_size, layer1_size)
        binary_len = np.dtype('float64').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float64')
        return word_vecs

def write_bin_vec_mikolov(vecs, fname_out):
    with open(fname_out, "wb") as f:
        vocab_size, layer1_size = len(vecs.items()), len(vecs.values()[0])
        print("Writing: %d %d" % (vocab_size, layer1_size));
        print("%d %d" % (vocab_size, layer1_size), file=f)
        binary_len = np.dtype('float64').itemsize * layer1_size
        for word, vector in vecs.items():
            print("%s " % word, file=f, end='')
            vector.astype('float64').tofile(f)
            print("", file=f)

def write_bin_vec_compressed(vecs, fname_out):
    with open(fname_out, "wb") as f:
        vocab_size, layer1_size = len(vecs.items()), len(vecs.values()[0])
        unique_values = np.unique(np.vstack(vecs.values()))
        n_bits = int(np.ceil(np.log(len(unique_values))))
        assert 2**n_bits >= len(unique_values)
        assert 8 % n_bits == 0
        print("%d %d %d" % (vocab_size, layer1_size, n_bits), file=f)
        n_codes_per_byte = 8 / n_bits
        value_to_code = {value : i for i, value in enumerate(unique_values)}
        print(value_to_code)
        sorted_values = [x[0] for x in sorted(value_to_code.items(), key=lambda x:x[1])]
        print("\n".join([str(x) for x in sorted_values]), file=f)
        for word, vec in vecs.items():
            print("%s " % word, end='', file=f)
            assert(len(vec) == layer1_size)
            for i in range(0, layer1_size, n_codes_per_byte):
                code = 0
                for j in range(n_codes_per_byte):
                    code <<= n_bits
                    code |= value_to_code[vec[i+j]]
                assert code < 256 and code >= 0
                f.write(chr(code))
            print("", file=f)
        
word_vecs = load_bin_vec(sys.argv[1])
stacked = np.vstack(word_vecs.values())
unique_values = np.unique(stacked)

print("Unique vector values: %s" % str(unique_values))
row_counts = np.sum(stacked, axis=0)
column_counts = np.sum(stacked, axis=1)

write_bin_vec_compressed(word_vecs, "out_compressed")
#write_bin_vec(word_vecs, "out_compressed")

