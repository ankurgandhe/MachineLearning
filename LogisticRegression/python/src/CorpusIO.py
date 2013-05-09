import sys 
import numpy 

def ReadTrainFile(filename):
    '''
    File should be in SVM format 
    '''
    Data=[]
    Labels =[]
    n_feats =0
    for l in open(filename):
        l=l.strip().split()
        label = int(l[0])
        if label==3:
            label=1
        else:
            label = 0
        Labels.append(label)
        feats,n_feats= ReadFeatures(l[1:],n_feats)
        Data.append(numpy.asarray(feats))
        
    return Data,Labels,n_feats

def ReadFeatures(splitline,n_feats):
    feats=[]
    i=1
    for w in splitline:
        w=w.strip().split(':')
        idx = int(w[0])
        f = float(w[1])
        if i<idx:
            while i<idx:
                feats.append(0.0)
                i=i+1
        feats.append(f)
        i=i+1
    if i < n_feats:
        while i==n_feats:
            feats.append(0.0)
    return feats,len(feats)
            
            
        
