import numpy
import scipy 
import math 
import sys

class LogisticRegression:
    ''' Class for training, predicting (test) and storing Logistic regression parameters 
        Input Data: Should be in SVM format : label 1:feat_val 2:feat_val ... 
        '''
    
    def __init__(self,bias=False,adaptive=False,epsilon=1e-5,eta=0.1,n_feat=0):
        self.W = []
        self.bias = bias             # bias 
        self.LabelMap = []           # LabelMap
        self.epsilon = epsilon       # convergence condition
        self.eta = eta               # learning rate 
        self.NumberFeats = n_feat    # Number of features 

    def AssignLabel(self,LabelMap):
        for l in LabelMap:
            self.LabelMap.append(l) #y=0,1,...
        #self.LabelMap.append(l2) #y=1
    def ConvertLabels(self,Y):
        NewY = []
        for y in Y:
            NewY.append(self.LabelMap.index(y))
        return NewY 
        
    def TrainLogisticRegression(self,X,Y,batch_size=-1,L1=0.0,L2=1.0):
        if L1==0:
            self.TrainLR_GradientAscent(X,Y,batch_size,L2)
        else:
            TrainLR_L1(self,X,Y,batch_size,L1,L2)
            
    def TrainLR_GradientAscent(self,DataX,DataY,batch_size,L2):
        ''' 
        DataX is Nxd : N samples , d features, List of numpy arrays
        DataY is Nx1 : sample labels, List  : Labels not modified to 0,1
        '''
	#Define parameters 
        eta0 = self.eta 
        L2lambda = L2
        if batch_size ==-1:
            batch_size = len(DataY)
        n_train_batches = len(DataY)/batch_size
	L2lambda = L2lambda / n_train_batches 
        DataY = self.ConvertLabels(DataY)

        self.b = 0 
        bias = 0
        if self.bias==True:
            bias = 1

	#initialize 
        self.W = numpy.zeros(self.NumberFeats)
        conv = False 
        iter= 0
        oldLogL = -1e10
        TotalError= numpy.zeros(self.NumberFeats)	

        while(not conv):
            iter = iter+1
            LogL = 0.0
            for i in range(n_train_batches):
                TotalError.fill(0)
                biasError = 0

                #Get batch error 
                for X,Y in zip(DataX[i*batch_size:(i+1)*batch_size], DataY[i*batch_size:(i+1)*batch_size]):
                    sumWx = numpy.dot(self.W,X) + self.b*bias
                    
                    probY0 = 1 / (1 + numpy.exp(sumWx));
                    probY1 = 1 - probY0
                    y_error  = Y - probY1;
                    LogL = LogL + numpy.log10(Y*probY1 + (1-Y)*probY0)
                    
                    TotalError = TotalError + X*y_error
                    biasError = biasError + bias*y_error
                
                #Update W 
                WNew = self.W + eta0 * (TotalError - L2lambda * self.W  )
                bNew = self.b + eta0 * (biasError - L2lambda * self.b )
                self.W = WNew
                self.b = bNew 
                
            DiffLogL = LogL - oldLogL
            
            if DiffLogL < self.epsilon:
                conv = True
            if iter % 10 ==0:
                print >> sys.stderr, iter,"...",
            
            oldLogL = LogL
        
        print >> sys.stderr, "Logistic regression training complete."
        print >> sys.stderr, self.W,self.b
    
    def TestLR(self,DataX):
        PredY=[]
        for X in DataX:
            yhat = self.PredictY(X)
            PredY.append(yhat)
        return PredY

    def PrintDetails(self,batch,L2):
        return 1 #print >> sys.stderr, batch,L1
    
    def PredictY(self,X):
        score = self.score(X)
        if score >0:
            return self.LabelMap[1]
        else:
            return self.LabelMap[0]
        
    def score(self,feat,bias=1):
        sumWx = numpy.dot(self.W,feat) + self.b*bias 
        return sumWx
