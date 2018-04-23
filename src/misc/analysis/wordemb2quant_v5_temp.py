import numpy as np
import time
import os
import scipy
from scipy import sparse, linalg
from scipy.sparse import coo_matrix
from scipy.stats import norm
import math
import multiprocessing
import os
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

path = '/lfs/1/tginart/word_em_expr/wiki_word_em_500.glove'
max_vec_len = 500


def columnwise_KM(nc,mi,inits,j,emb):
    kmeans = KMeans(n_clusters=nc,max_iter=mi,n_init=inits,n_jobs=1).fit(emb[:,j].reshape(-1,1))
    return np.concatenate(kmeans.cluster_centers_[kmeans.labels_])


def quantize(U,S,bits_per_entry,n_cores):
    N = len(S)
    bit_allot = clever_round(compute_bit_allot(singvals,bits_per_entry,0.1))
    with Parallel(n_jobs=n_cores) as parafor:
        U_q = parafor(delayed(columnwise_KM)(2**bit_allot[j],200,2,j,U) for j in range(0,N))
        U_q = np.transpose(U_q)
        return U_q,U_q*S




def columnwise_KM_preloaded(emb_KM,bits_per_entry, column_num):
    #assumes the emb_KM files are loaded!!!!!!!!!!!!
    return emb_KM[bits_per_entry,column_num,:]
    
def load_KM_embs():
    emb_KM0 = np.load('emb_KM0.npy')
    emb_KM1 = np.load('emb_KM1.npy')
    emb_KM2 = np.load('emb_KM2.npy')
    emb_KM3 = np.load('emb_KM3.npy')
    emb_KM4 = np.load('emb_KM4.npy')
    emb_KM5 = np.load('emb_KM5.npy')
    return np.array([emb_KM0,emb_KM1,emb_KM2,emb_KM3,emb_KM4,emb_KM5])

    

def load_sing_vals(path):
    string_array = open(path, 'r').read().split()
    return np.array(string_array).astype(np.float)

def write_emb(emb,path):
    #TODO: this fucntion is sketchy
    f = open(path,'w')
    line = 'someword'
    for x in emb:
        for y in x:
            line += ' '
            line += str(y)
        line += '\n'
    f.write(line)
    return
    
    
 
def load_emb(path, max_vec_len):
    
    f = open(path, 'r')
    dat = [_.strip() for _ in f]
    if len(dat[0].split()) == 2: dat = dat[1:]
    dim = len(dat[0].split()) - 1
    if(max_vec_len > -1):
        dim = min(dim, max_vec_len)
    m = np.zeros((len(dat), dim))
    vocab = []
    cnt = 0
    for i, _ in enumerate(dat):
        d = _.split(" ")
        if (len(d) != dim + 1):
            cnt += 1
        w = d[0]
        v = d[-dim:]
        m[i] = v
        vocab.append(w)
    return m,vocab

#emb,vocab = load_emb(path, max_vec_len)

def custom_quantize(vect, bit_allot):
    mu = np.mean(vect)
    if(bit_allot < 0.001):
        return np.zeros(vect.shape).fill(mu)
    
    if(bit_allot < 1.001):
        for v,i in vect:
            if (math.abs(v-mu) < math.abs(v-0)):
                vect[i] = mu
            else:
                vect[i] = 0

        return vect
    
    if(bit_allot < 2.001):
        std = np.std(vect)
        quanta = np.array([mu,0,std,-std])
        for i in range(len(vect)):
            v = vect[i]
            argmin = np.argmin([np.abs(v-mu),np.abs(v-0),np.abs(v-std),np.abs(v+std)])
            vp = quanta[argmin]
            vect[i] = vp
        return vect
    if(bit_allot < 3.001):
        return vect
    return vect



def uniform_quantize(vect, bit_allot):
    if(bit_allot <= 1):
        return np.zeros(vect.shape)
    L = 2**bit_allot-1 
    a = max(np.max(vect), -1*np.min(vect))+1e-7
    delta = 2*a/(L-1)
    return np.round((vect+a)/delta)*delta-a

def uniform_quantize_old(vect, bit_allot):
#takes bit allotment per element in column vector
    L = 2**bit_allot
    delta = 1/L
    return np.floor((vect+1)/2*L)/L*2-1+delta

#def renorm_midriser(vect, bit_allot):
#    L = 2**bit_allot
#    deilta = 

def gaussian_quantize_root3(col_vect, bit_allot):
    mean = np.mean(col_vect)
    var = np.var(col_vect)
    std_dev = np.sqrt(var)
    psi_vect = 2*np.clip(norm.cdf(col_vect,mean,std_dev*np.sqrt(3)),-1+1e-10,1-1e-10)-1
    psi_vect = 0.5*(uniform_quantize_old(psi_vect,bit_allot)+1)
    return np.clip(norm.ppf(psi_vect,mean,std_dev*np.sqrt(3)),-1,1)


def gaussian_quantize2(col_vect, bit_allot):
    mean = np.mean(col_vect)
    var = np.var(col_vect)
#    L = 2**bit_allot
#    deilta = 

def gaussian_quantize_root3(col_vect, bit_allot):
    mean = np.mean(col_vect)
    var = np.var(col_vect)
    std_dev = np.sqrt(var)
    psi_vect = 2*np.clip(norm.cdf(col_vect,mean,std_dev*np.sqrt(3)),-1+1e-10,1-1e-10)-1
    psi_vect = 0.5*(uniform_quantize_old(psi_vect,bit_allot)+1)
    return np.clip(norm.ppf(psi_vect,mean,std_dev*np.sqrt(3)),-1,1)


def gaussian_quantize2(col_vect, bit_allot):
    mean = np.mean(col_vect)
    var = np.var(col_vect)
    sqrtvar = np.sqrt((np.sqrt(var)))
    psi_vect = 2*np.clip(norm.cdf(col_vect,mean,sqrtvar),-1+1e-10,1-1e-10)-1
    psi_vect = 0.5*(uniform_quantize_old(psi_vect,bit_allot)+1)
    #print(psi_vect)
    return  np.clip(norm.ppf(psi_vect,mean,sqrtvar),-1,1)



def gaussian_quantize(col_vect, bit_allot):
    mean = np.mean(col_vect)
    var = np.var(col_vect)
    sqrtvar = (np.sqrt(var))
    psi_vect = 2*np.clip(norm.cdf(col_vect,mean,sqrtvar),-1+1e-10,1-1e-10)-1
    psi_vect = 0.5*(uniform_quantize_old(psi_vect,bit_allot)+1)
    #print(psi_vect)
    return  np.clip(norm.ppf(psi_vect,mean,sqrtvar),-1,1)

def gaussian_quantize3(col_vect, bit_allot):
    n = len(col_vect)
    psi_vect = 2*norm.cdf(col_vect,0,1/n,-1+1e-10,1-1e-10)-1
    psi_vect = 0.5*(uniform_quantize(psi_vect,bit_allot)+1)
    return  np.clip(norm.ppf(psi_vect,0,1/n),-1,1)

def compute_bit_allot(var_vect, bit_budget, err_tol):
#computes bit allotment, needs variances as input 
    total_budget = len(var_vect)*bit_budget          
    lamb_max = max(var_vect)
    lamb_min = 1e-100
    rate = 0
    while(np.abs(rate-total_budget) > err_tol):
        lamb = (lamb_max - lamb_min)/2 + lamb_min
        rate = 0.5*sum(np.log2(var_vect/np.minimum(var_vect,lamb)))
        if(rate > total_budget):
            lamb_min = lamb
        else:
            lamb_max = lamb

    return 0.5*np.log2(var_vect/np.minimum(var_vect,lamb))

def renorm_uni_round(bit_allot_vect):
    bit_bank = 0
    n = len(bit_allot_vect)
    for i,a in reversed(list(enumerate(bit_allot_vect))):
        a_up = np.ceil(a)
        a_dwn = np.floor(a)
        diffup = a_up - a
        diffdwn = a - a_dwn
        if(a < 2.1):
            bit_bank += a
            bit_allot_vect[i] = 0
        elif(bit_bank > diffup):
            bit_bank -= diffup
            bit_allot_vect[i] = a_up
        else:
            bit_bank += diffdwn
            bit_allot_vect[i] = a_dwn

        #print(bit_bank)
    i = 0
    while(bit_bank > 1):
        if(bit_allot_vect[i] > 0.00001):
            bit_allot_vect[i] += 1
            bit_bank -= 1
        i += 1
        i = i%n
       # print(i)
    return bit_allot_vect



def clever_round(bit_allot_vect):
#rounds the bit allot vect such that <1 bit goes unused in aggregate
    bit_bank = 0
    for i,a in enumerate(bit_allot_vect):
        a_up = np.ceil(a)
        a_dwn = np.floor(a)
        diffup = a_up - a
        diffdwn = a - a_dwn
        if(bit_bank > diffup):
            bit_bank -= diffup
            bit_allot_vect[i] = a_up
        
        else:
            bit_bank += diffdwn
            bit_allot_vect[i] = a_dwn

    return bit_allot_vect
 


def cleverer_round(bit_allot_vect):
#rounds the bit allot vect such that <1 bit goes unused in aggregate
    bit_bank = 0
    for i,a in reversed(list(enumerate(bit_allot_vect))):
        a_up = np.ceil(a)
        a_dwn = np.floor(a)
        diffup = a_up - a
        diffdwn = a - a_dwn
        bit_bank_old = bit_bank
        #print(bit_bank)
        if(bit_bank > diffup):
            bit_bank -= diffup
            bit_allot_vect[i] = a_up
        
        else:
            bit_bank += diffdwn
            bit_allot_vect[i] = a_dwn
        if(bit_allot_vect[i] <= 1.000000001):
            #print(i)
            bit_allot_vect[i] = 0
            bit_bank = bit_bank_old
            bit_bank += a

    if (bit_bank > 1):
        #print(bit_bank)
        bit_allot_vect[0] += np.floor(bit_bank)
        #bit_bank -= np.floor(bit_bank)  
    #print(bit_bank)
    return bit_allot_vect

def write2file_FP(emb,precision,total_budget,filepath):
    num_cols = int(np.floor(total_budget/(len(emb[:,0])*precision)))
    if(precision == 16):
        cheap_emb = np.float16(emb[:,0:num_cols])
    elif(precision == 32):
        cheap_emb = np.float32(emb[:,0:num_cols])
    else:
        cheap_emb = np.float64(emb[:,0:num_cols])
