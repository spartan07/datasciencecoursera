import numpy as np
def sigma(x,deriv=False):
    sig=1/(1+np.exp(-x))    
    if deriv==True:
		return sig*(1-sig)
    return sig 


#Input Data
#X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
X=np.array([[1,0],[0,0],[1,1]])
y=np.array([[0],[1],[1]])

np.random.seed(1)

syn0=np.random.random((2,10))
syn1=np.random.random((10,1))

for j in xrange(60000):
    l0=X
    l1=sigma(np.dot(X,syn0))
    l2=sigma(np.dot(l1,syn1))
    
    err=y-l2
    if j%10000==0:
        tr=np.mean(np.abs(err))
        print "Error %f" %tr
    
    l2delta=err*sigma(l2,deriv=True)
    l1err=np.dot(l2delta,syn1.T)
    l1delta=l1err*sigma(l1,deriv=True)
    
    syn1+=l1.T.dot(l2delta)
    syn0+=l0.T.dot(l1delta)

print "Output"
print l2    



class Nnet():
    def __init__(self):
        self.inpsize=2
        self.outsize=1
        self.hidsize=3
        
        self.w1=np.random.random((self.inpsize,self.hidsize))
        self.w2=np.random.random((self.hidsize,self.outsize))
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def forward(self,X):
        self.z2=np.dot(X,self.w1)
        self.a2=self.sigmoid(self.z2)
        self.z3=np.dot(self.a2,self.w2)
        yout=self.sigmoid(self.z3)
        return yout
    
    def sigmoid_deriv(self,X):
        return (np.exp(-X)/np.power((1+np.exp(-X)),2))
    
    def cost(self,X,y):
        yhat=self.forward(X)
        J=np.mean(0.5*(y-yhat)**2)
        return J
    
    def costFunc(self,X,y):
        self.yout=self.forward(X)
        delta3=np.multiply(-(y-self.yout),self.sigmoid_deriv(self.z3))
        self.dJdW2=np.dot(self.a2.T,delta3)
        
        delta2=np.dot(delta3,self.w2.T)*self.sigmoid_deriv(self.z2)
        self.dJdW1=np.dot(X.T,delta2)
        return self.dJdW1,self.dJdW2

NN=Nnet()   
yhat=NN.forward(X)  
dJdW1,dJdW2=NN.costFunc(X,y) 
c1=NN.cost(X,y) 



