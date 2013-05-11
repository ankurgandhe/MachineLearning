import sys 
sys.dont_write_bytecode = True
from LogisticRegression import *
from CorpusIO import *
import argparse
from Evaluation import PredictionAccuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainfile", type=str, help="Training File in SVM format")
    parser.add_argument("-test", "--testfile", type=str, help="Testing File in SVM format")
    parser.add_argument("-l1", "--l1reg", type=float, default='0.00', help="Coeff of L1-reg")
    parser.add_argument("-l2", "--l2reg", type=float, default='1', help="Coeff of L2-reg")
    parser.add_argument("-batch", "--batchsize", type=int, default=1, help="Batch size")
    parser.add_argument("-b", "--bias", type=int, default=0, help="Add bias to the regression")
    parser.add_argument("-r", "--learningrate", type=float, default='0.1', help="Learning rate for gradient ascent")
    parser.add_argument("-e", "--convergence", type=float, default='1e-4', help="Convergence condition")
    parser.add_argument("-v", "--verbosity", type=int, default=0, help="Verbosity level")
    parser.add_argument("-z", "--algorithm", type=str, default="gd", help="Algorithm to be used: gd : gradient descent, cg:conjugate descent")

    args = parser.parse_args()
    if not args.trainfile:	
	print >> sys.stderr, "Training file needs to be provided. Use -h or --help for more options"	
	sys.exit()
    trainFile = args.trainfile
    L1_reg = args.l1reg
    L2_reg = args.l2reg
    batch_size = args.batchsize
    b = bool(args.bias)
    learning_rate = args.learningrate
    epsilon = args.convergence
    algorithm = args.algorithm
    
    Data,labels,LabelMap,n_feats = ReadFeatureFile(trainFile)

    LR = LogisticRegression(n_feat=n_feats,epsilon=epsilon,eta=learning_rate,bias=b)
    LR.AssignLabel(LabelMap)

    LR.TrainLogisticRegression(X=Data,Y=labels,batch_size=batch_size,L1=L1_reg,L2=L2_reg,algo=algorithm)
    if args.testfile:
        TestData,Testlabels,labelMap,nfeat = ReadFeatureFile(args.testfile)
        Y = LR.TestLR(TestData)
        print PredictionAccuracy(actual=Testlabels,predicted=Y)
        
