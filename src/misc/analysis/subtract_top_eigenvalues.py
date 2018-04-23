import sys
from sklearn.decomposition import PCA
from convert_word2bits import *

def subtract_top_eigenvalues(fname_in, fname_out):
    vec = load_vec(fname_in)
    values = np.stack(vec.values())

    pca = PCA()
    pca.fit(values)

    print("Computing components...")
    components = pca.components_[:,5:]
    print("Done computing components...")
    print("PCA...")
    new_values = values.dot(components.dot(components.T))
    print("Done PCA...")

    new_vec = {k:new_values[i,:] for i, (k, v) in enumerate(vec.items())}
    write_bin_vec_text(new_vec, fname_out)
    
    return values


subtract_top_eigenvalues("/lfs/1/maxlam/automate_output_vectors/vectors_binary=0_bitlevel=0_iter=10_min-count=5_negative=12_reg=.001_sample=0.0001_size=200_threads=50_train=dfsscratch0maxlamwiki.en.txt_window=10", "/lfs/1/maxlam/vectors_binary=0_bitlevel=0_iter=10_min-count=5_negative=12_reg=.001_sample=0.0001_size=200_threads=50_train=dfsscratch0maxlamwiki.en.txt_window=10_allbuttop5")
    
