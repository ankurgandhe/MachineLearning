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

    def ConvertLabels(self,Y):
        NewY = []
        for y in Y:
            idx = self.LabelMap.index(y) # 0 -> -1 , 1 -> 1 
            if idx==0:
                idx=-1
            NewY.append(idx)
        return NewY 
        
    def TrainLogisticRegression(self,X,Y,batch_size=-1,L1=0.0,L2=1.0,algo="gd"):
        if len(self.LabelMap)==2:
            if L1==0:
                self.TrainLR(X,Y,batch_size,L2,algo)
            else:
                self.TrainLR_L1(self,X,Y,batch_size,L1,L2)
        else: #Multivariate Case
            if L1==0:
                self.MultiVariateTrainLR(X,Y,batch_size,L2,algo)

            
    def MultiVariateTrainLR(self,DataX,DataY,batch_size,L2,algo):
        self.SetW = []
	#Train multiple classifiers 
        for i in range(0,len(self.LabelMap)):
            
            tmp = self.LabelMap[0]
            self.LabelMap[0]=self.LabelMap[i]
            self.LabelMap[i]=tmp 

            self.TrainLR(DataX,DataY,batch_size,L2,algo)
            self.SetW.append(self.W)

            self.LabelMap[i]=self.LabelMap[0]
            self.LabelMap[0]=tmp
            
            print >> sys.stderr, "Trained Classifier no", i


    def TrainLR(self,DataX,DataY,batch_size,L2,algo):
        ''' 
        DataX is Nxd : N samples , d features, List of numpy arrays
        DataY is Nx1 : sample labels, List  : Labels not modified to 0,1
        '''
	#Define parameters 
        eta0 = self.eta 
        
        L2lambda = L2
        if batch_size ==-1 or algo=="cg":
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
        
        while(not conv):
            iter = iter+1
            LogL = 0.0
            u_old = 0 #For CG
            g_old = 0 #For CG
            for i in range(n_train_batches):
                #Get batch error 
                if algo=="gd": #Gradient Descent
                    LogL = LogL + self.update(DataX[i*batch_size:(i+1)*batch_size], bias, DataY[i*batch_size:(i+1)*batch_size],eta0,L2lambda)
                else: #Conjugate Gradient
                    newLogL,u_old,g_old = self.update_CG(DataX[i*batch_size:(i+1)*batch_size], bias, DataY[i*batch_size:(i+1)*batch_size],eta0,L2lambda,u_old,g_old)
                    LogL = LogL + newLogL 

            DiffLogL = LogL - oldLogL
            
            if numpy.abs(DiffLogL) < self.epsilon:
                conv = True
            if iter % 10 ==0:
                print >> sys.stderr, iter,"...",
            if iter > 2000:
	        print >> sys.stderr, "Done", iter, " iterations. Try changing learning rate via -r option, or use -z cg for conjugate gradient descent"
	        conv=True
            oldLogL = LogL
        
        print >> sys.stderr, "Logistic regression training complete."
        print >> sys.stderr, self.W,self.b
    
    def update(self,DataX,biasX,DataY,eta0,L2lambda): #Returns LogL 
        LogL=0
        TotalError= numpy.zeros(self.NumberFeats)
	
        biasError = 0

        for X,Y in zip(DataX, DataY):
            if Y > 1:
                Y=1
            sumWx = numpy.dot(self.W,X) + self.b*biasX
            #if Y==0:
            #    Y=-1
            #probY0 = 1 / (1 + numpy.exp(sumWx));
            #probY1 = 1 - probY0
            #y_error  = Y - probY1;
            #LogL = LogL + numpy.log10(Y*probY1 + (1-Y)*probY0)
            LogL = LogL - numpy.log10(1+numpy.exp(-Y*sumWx))
            TotalError = TotalError + X*Y*(1.-self.sigmoid(Y*sumWx))#y_error
            biasError = biasError + biasX*Y*(1.-self.sigmoid(Y*sumWx))#y_error
            
        #Update W
        self.W = self.W + eta0 * (TotalError - L2lambda * self.W  )
        self.b = self.b + eta0 * (biasError - L2lambda * self.b )
        
        return LogL
 
    def update_CG(self,DataX,biasX,DataY,eta0,L2lambda,u_old,g_old):
        
        g = -L2lambda*self.W 
        A =[]
        LogL=0
        for X,Y in zip(DataX, DataY):
            sumWx = numpy.dot(self.W,X) + self.b*biasX
            LogL = LogL - numpy.log10(1+numpy.exp(-Y*sumWx))
            g = g + X*Y*(1.-self.sigmoid(Y*sumWx))
            sigWX = self.sigmoid(sumWx)
            A.append(sigWX*( 1 - sigWX))
            
        if u_old == 0:
            u = g
        else:
            beta = numpy.dot(g,(g - g_old)) / numpy.dot(u_old,(g - g_old))
            u = g - u_old * beta 
            
        denW = L2lambda*numpy.dot(u,u) 
        for aii,X,Y in zip(A,DataX,DataY):
            denW = denW + aii*numpy.square(numpy.dot(u,X))
        
        self.W = self.W + ( numpy.dot(g,u) / denW ) * u 

        return LogL , u_old, g_old
        
    def TestLR(self,DataX):
        PredY=[]
        for X in DataX:
            yhat = self.PredictY(X)
            PredY.append(yhat)
        return PredY


    def PrintDetails(self,batch,L2):
        return 1 #print >> sys.stderr, batch,L1
    
    def PredictY(self,X):
        if len(self.LabelMap)==2:
            score = self.score(self.W,X)
            if score >0:
                return self.LabelMap[1]
            else:
                return self.LabelMap[0]
        else:
            Scores=[]
            for idx,W in enumerate(self.SetW):
                Scores.append((self.score(W,X),idx))
            Scores = sorted(Scores)
            return self.LabelMap[Scores[0][1]]
        
    def score(self,W,feat,bias=1):
        sumWx = numpy.dot(W,feat) + self.b*bias 
        return sumWx

    def sigmoid(self,x):
        return 1./(1. + numpy.exp(-x))
