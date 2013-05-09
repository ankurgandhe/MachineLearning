import numpy
import scipy 
import math 

class LogisticRegression:
    ''' Class for training, predicting (test) and storing Logistic regression parameters 
        Input Data: Should be in SVM format : label 1:feat_val 2:feat_val ... 
        '''
    
    def __init__(self,bias=False,adaptive=False,epsilon=1e-4,eta=1e-2,n_feat=0):
        self.W = []
        self.bias = bias             # bias 
        self.LabelMap = {}           # LabelMap
        self.epsilon = epsilon       # convergence condition
        self.eta = eta               # learning rate 
        self.NumberFeats = n_feat    # Number of features 

    def AssignLabel(self,l1,l2):
        LabelMap.append(l1) #y=0
        LabelMap.append(l2) #y=1
    
    def TrainLogisticRegression(self,X,Y,batch_size=-1,L1=0.0,L2=1):
        if L1==0:
            TrainLR(self,X,Y,batch_size,L2)
        else:
            TrainLR_L1(self,X,Y,batch_size,L1,L2)
            
    def TrainLR(self,DataX,DataY,batch_size,L2):
        ''' 
        DataX is Nxd : N samples , d features, List of numpy arrays
        DataY is Nx1 : sample labels, List  : Labels already modified to 0,1
        '''

        eta0 = self.eta 
        L2lambda = L2
        if batch_size ==-1:
            batch_size = len(DataY)
        n_train_batches = len(DataY)/batch_size
        self.b = 0 
        bias = 0
        if self.bias==True:
            bias = 1
        self.W = numpy.zeros((self.NumberFeats,1))
        conv = False 
        iter= 0
        Wold = self.W 
        while(!conv):
            for i in range(n_train_batches):
                TotalError= numpy.asarry((self.NumberFeats))
                biasError = 0
                #Get batch error 
                for X,Y in DataX[i*batch_size:(i+1)*batch_size], Y = DataY[i*batch_size:(i+1)*batch_size]):
                    sumWx = numpy.sum(self.W*X) + self.b*bias
                    probY0 = sumWx > 100 ? 0 : 1 / (1 + numpy.exp(sumWx));
                    yl_prob = Y - (1 - probY0);
                    TotalErorr = TotalError + X*yl_prob
                    biasError = biasError + bias*yl_prob

                #Update W 
                WNew = self.W + eta0 * (TotalError - L2lambda * self.W )
                bNew = self.b + eta0 * (biasError - L2lambda * self.b )
                self.W = WNew
            DiffW = numpy.amax(numpy.absolute(WNew - Wold))
            if DiffW < self.epsilon:
                break
            if iter % 10 ==0:
                print >> sys.stderr, iter,"...",
            Wold = WNew 
        
        print >> sys.stderr, "Logistic regression training complete."
        print >> sys.stderr, self.W,self.b
    
                
