import sys 
sys.dont_write_bytecode = True
from LogisticRegression import *
from CorpusIO import *

trainFile = sys.argv[1]
Data,labels,n_feats = ReadTrainFile(trainFile)

LR = LogisticRegression(n_feat=n_feats,epsilon=1e-10)
LR.TrainLogisticRegression(X=Data,Y=labels,batch_size=1)


