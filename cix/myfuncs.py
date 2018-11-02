import chemfp
from chemfp import search
import pandas as pd
import time
import sys
import subprocess as sp
from rdkit import Chem
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import pylab
import random
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold as ms
from rdkit.Chem.Draw import IPythonConsole


### The results of the Taylor-Butina clustering
class ClusterResults(object):
    def __init__(self, true_singletons, false_singletons, clusters):
        self.true_singletons = true_singletons
        self.false_singletons = false_singletons
        self.clusters = clusters



### The clustering implementation
def taylor_butina_cluster(similarity_table):
    # Sort the results so that fingerprints with more hits come
    # first. This is more likely to be a cluster centroid. Break ties
    # arbitrarily by the fingerprint id; since fingerprints are
    # ordered by the number of bits this likely makes larger
    # structures appear first.:

    # Reorder so the centroid with the most hits comes first.  (That's why I do
    # a reverse search.)  Ignore the arbitrariness of breaking ties by
    # fingerprint index

    centroid_table = sorted(((len(indices), i, indices)
                                 for (i,indices) in enumerate(similarity_table.iter_indices())),
                            reverse=True)

    # Apply the leader algorithm to determine the cluster centroids
    # and the singletons:

    # Determine the true/false singletons and the clusters
    true_singletons = []
    false_singletons = []
    clusters = []

    seen = set()
    for (size, fp_idx, members) in centroid_table:
        if fp_idx in seen:
            # Can't use a centroid which is already assigned
            continue
        seen.add(fp_idx)

        # Figure out which ones haven't yet been assigned
        unassigned = set(members) - seen

        if not unassigned:
            false_singletons.append(fp_idx)
            continue

        # this is a new cluster
        clusters.append((fp_idx, unassigned))
        seen.update(unassigned)

    # Return the results:
    return ClusterResults(true_singletons, false_singletons, clusters)



### Calculate distance matrix for hierarchical clustering
def distance_matrix(arena):
    n = len(arena)

    # Start off a similarity matrix with 1.0s along the diagonal
    similarities = np.identity(n, "d")

    ## Compute the full similarity matrix.
    # The implementation computes the upper-triangle then copies
    # the upper-triangle into lower-triangle. It does not include
    # terms for the diagonal.
    results = search.threshold_tanimoto_search_symmetric(arena, threshold=0.0)

    # Copy the results into the NumPy array.
    for row_index, row in enumerate(results.iter_indices_and_scores()):
        for target_index, target_score in row:
            similarities[row_index, target_index] = target_score

    # Return the distance matrix using the similarity matrix
    return 1.0 - similarities



### Create a smis list from a smiles file (just smiles)
def smif2smis(name):
    smidf = pd.read_csv(name, delim_whitespace = True, names = ['smiles'], header = None)
    return list(smidf['smiles'])



### Find the correct smiles in a list of smiles
def corrsmis(smis):
    n = len(smis)
    corr_smi_yn = [x != None for x in [Chem.MolFromSmiles(s) for s in smis]]
    ncorr = sum(corr_smi_yn)
    smis = [smi for i, smi in enumerate(smis) if corr_smi_yn[i] == True]
    wrongsmis = [smi for i, smi in enumerate(smis) if corr_smi_yn[i] == False]
    return ncorr, n, smis, wrongsmis



### Create a dataframe of smiles, id from smiles list
def smis2smidf(smis):
    return pd.DataFrame({'smiles': smis, 'id': ['s' + str(x) for x in range(1, len(smis)+1)]}, columns = ['smiles','id'])



### Create a dataframe of smiles, id from smiles file
def smisf2smidf(smisf, noid = True):
    
    if noid:
        smidf = pd.read_csv(smisf, delim_whitespace = True, names = ['smiles'], header = None)
    else:
        smidf = pd.read_csv(smisf, delim_whitespace = True, names = ['smiles','id'], header = None)
    
    return smidf



### Create arena from smiles df
def smidf2arena(smidf):
    # Write df of smiles, id
    smidf.to_csv('smidf.smi', header = False, sep = ' ', index = False)
    
    # Generate fps file
    sp.call(['rdkit2fps', './smidf.smi', '-o', 'smidf.fps'])
    
    ## Load the FPs into an arena
    try:
        arena = chemfp.load_fingerprints('./smidf.fps')
    except IOError as err:
        sys.stderr.write("Cannot open fingerprint file: %s" % (err,))
        raise SystemExit(2)
    
    # Remove files
    sp.call(['rm', './smidf.smi', './smidf.fps'])
    
    # Return arena
    return arena



### Cluster from smiles df
def clusmidf(smidf, th = 0.8, method = 'butina', arena = None):
    
    if method != 'butina' and method != 'cl':
        print('Please select butina or cl')
        return None
        
    # Init time counter
    start = time.time()
    
    # Get the arena
    if arena is None:
        arena = smidf2arena(smidf)

    # Do the clustering
    if method == 'butina':
        # Generate the similarity table
        similarity_table = search.threshold_tanimoto_search_symmetric(arena, threshold = th)
    
        # Cluster the data
        clus_res = taylor_butina_cluster(similarity_table)
        
        # Output
        out = []
        # We need to re-sort the clusters as the creation of them does not generate a monotonously decreasing list
        cs_sorted = sorted([(len(c[1]), c[1], c[0]) for c in clus_res.clusters], reverse = True)
        for i in range(len(cs_sorted)):
            cl = []
            c = cs_sorted[i]
            cl.append(arena.ids[c[2]]) # Retrieve the arenaid of the centroid and add to the cluster
            cl.extend([arena.ids[x] for x in c[1]]) # Retrieve the arenaid of the neighbors and add to cluster
            out.append(cl)
        for i in range(len(clus_res.false_singletons)):
            cl = [arena.ids[clus_res.false_singletons[i]]]
            out.append(cl)
        for i in range(len(clus_res.true_singletons)):
            cl = [arena.ids[clus_res.true_singletons[i]]]
            out.append(cl)
        
    elif method == 'cl':
        # Generate the condensed distance table
        distances  = ssd.squareform(distance_matrix(arena))
        
        # Cluster the data
        clus_res = fcluster(linkage(distances, method='complete'), th, 'distance')
        
        # Ouptut
        aids = arena.ids
        out = []
        for i in np.unique(clus_res):
            cl = [aids[i] for i in list(np.where(clus_res == i)[0])]
            out.append(cl)
        out = [x[2] for x in sorted([(len(x), i, x) for (i, x) in enumerate(out)], reverse = True)]


    # End time count and report
    end = time.time()
    elapsed_time = end - start
    print('Clustering time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # Return cluster results
    return out
    
    

### Draw a set of molecules from smiles list
def paintmols(smis, molsPerRow = 5, subImgSize=(150,150)):
    ms = [Chem.MolFromSmiles(s) for s in smis]
    return Draw.MolsToGridImage(ms,molsPerRow=molsPerRow,subImgSize=subImgSize)



### Diversity analysis
def divan(smidf, summ = False, OnlyBu = False, arena = None):
    
    start = time.time()
    
    # Cluster by butina and cl
    clr_bu = clusmidf(smidf, arena = arena)
    if(not OnlyBu):
        clr_cl = clusmidf(smidf, method = 'cl', th = 0.55, arena = arena)
    
    # Count the number of cluster in each method
    ncl_bu = len(clr_bu)
    if(not OnlyBu):
        ncl_cl = len(clr_cl)
    
    # Count Murko frameworks
    fras = list(set([Chem.MolToSmiles(ms.GetScaffoldForMol(Chem.MolFromSmiles(s))) for s in smidf.smiles]))
    nfra = len(fras)
    
    # Count generic Murko frameworks
    frasg = list(set([Chem.MolToSmiles(ms.MakeScaffoldGeneric(Chem.MolFromSmiles(s))) for s in smidf.smiles]))
    nfrag = len(frasg)
    
    end = time.time()
    eltime = end - start
    print('Diversity analysis time: ' + time.strftime("%H:%M:%S", time.gmtime(eltime)))
    
    if(summ):
        if(OnlyBu):
            return ncl_bu, nfra, nfrag
	else:
    	    return ncl_bu, ncl_cl, nfra, nfrag
    else:
        if(OnlyBu):
            return clr_bu, fras, frasg
	else:
	    return clr_bu, clr_cl, fras, frasg



### Novelty analysis
def novan(smidfq, smidft, th = 0.7):
    
    start = time.time()
    
    # Get the arenas
    arq = smidf2arena(smidfq)
    art = smidf2arena(smidft)
    
    end = time.time()
    eltime = end - start
    print('Arenas creation time: ' + time.strftime("%H:%M:%S", time.gmtime(eltime)))
    
    # Find hits
    results = chemfp.search.threshold_tanimoto_search_arena(arq, art, threshold=th)
    
    # Generate list with new guys (no neighbors in target arena) and calculate its length
    news = []
    for query_id, query_hits in zip(arq.ids, results):
        if len(query_hits) == 0:
            news.append(query_id)
    
    
    # Generate list of frameworks for query and target
    fraq = [Chem.MolToSmiles(ms.GetScaffoldForMol(Chem.MolFromSmiles(s))) for s in smidfq.smiles]
    fraq = list(np.unique(fraq))
    frat = [Chem.MolToSmiles(ms.GetScaffoldForMol(Chem.MolFromSmiles(s))) for s in smidft.smiles]
    frat = list(np.unique(frat))
    
    newfraqs = [f for f in fraq if f not in frat]
    
    # Generate list of generic frameworks for query and target
    gfraq = [Chem.MolToSmiles(ms.MakeScaffoldGeneric(Chem.MolFromSmiles(s))) for s in smidfq.smiles]
    gfraq = list(np.unique(gfraq))
    gfrat = [Chem.MolToSmiles(ms.MakeScaffoldGeneric(Chem.MolFromSmiles(s))) for s in smidft.smiles]
    gfrat = list(np.unique(gfrat))
    
    newgfraqs = [f for f in gfraq if f not in gfrat]
    
    
    end = time.time()
    eltime = end - start
    print('Novelty analysis time: ' + time.strftime("%H:%M:%S", time.gmtime(eltime)))
    
    return news, fraq, newfraqs, gfraq, newgfraqs


### Plot clusters
def plotclus(d, xlab, ylab, xloglab, yloglab):

    ax1 = plt.axes()  # standard axes
    ax2 = plt.axes([0.45, 0.45, 0.4, 0.4])
    ax1.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(xloglab)
    ax2.set_ylabel(yloglab)
    ax2.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)
