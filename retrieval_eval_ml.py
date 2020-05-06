from heapq import nlargest
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.io as sio
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score, precision_recall_curve

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def knn(k, list):
    # Returning a list of top K-Nearest-Neighbours
    if k == 1:
        m = max(list)
        return [i for i, j in enumerate(list) if j == m] # In case there is more than one best score
    else:
        list_ = nlargest(k, enumerate(list), itemgetter(1))
        return [idx for idx,_ in list_]

def cosine_similarity_recall_k(view1, view2, k):
    ## Finding the top K-Nearest-Neighbours by the cosine metric ##
    if not isinstance(k, int) or k < 1:
        'k must be an integer larger than zero'
        return
    hits = 0
    res_idxs = {}
    n = len(view1)
    for idx in range(n):
        sample = [view1[idx]]
#        cosine_distances = cosine_similarity(sample, view2)[0]
        cosine_distances = np.dot(sample, view2.transpose())[0]
        result = knn(k,cosine_distances)
        if k > 1 or len(result) == 1:
            res_idxs[idx] = result
        elif idx in result and k == 1:
            res_idxs[idx] = [idx] # We would like to return only one index for k = 1
        else:
            res_idxs[idx] = [result[0]] # We would like to return only one index for k = 1
        # Checking if we got a hit
        if idx in result and len(result) <= k:
            hits += 1
    total = idx + 1
    recall = float(hits) / (total)
    return res_idxs, recall, hits, total

## Test ##
def evaluate(F, G, cfg):
    F = F.data.numpy()
    G = G.data.numpy()
    feat_F = cfg.feats[0]
    feat_G = cfg.feats[1]
    
    ########## map
    # Compute similarity matrix for each cross-modal pair
    IT_similarity = np.inner(F,G) / ((F**2).sum(axis=1)**.5 + ((G**2).sum(axis=1)**.5)[np.newaxis])
    # read ground truth ()
    label = loadmat('data.mat') #cnn
    test_labels = label['label_te']
 
    all = []
    for i in range(len(test_labels)):   
        gnd = np.dot(test_labels, test_labels[i])           
        ground_truth_label = np.where(gnd > 0,1,0)                
        dists = IT_similarity[i]
        score = average_precision_score(ground_truth_label, dists)    
        all.append(score)
    print('MAP' + ' - ' + feat_F + ' to ' + feat_G + ':')
    F_map = np.mean(all)
    print(F_map)

    
    # Text to images queries (transpose the similarity matrix)
    all = []
    for i in range(len(test_labels)):   
        gnd = np.dot(test_labels, test_labels[i])           
        ground_truth_label = np.where(gnd > 0,1,0) 
        dists = IT_similarity.T[i]               
        score = average_precision_score(ground_truth_label, dists)
        all.append(score)
    print('MAP' + ' - ' + feat_G + ' to ' + feat_F + ':')
    G_map = np.mean(all)
    print(G_map)  
    
    F_map = round(F_map,4)
    G_map = round(G_map,4)
    return F_map, G_map   
