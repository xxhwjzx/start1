# -- encoding:utf-8 --
import numpy as np

class Neuralnetwork:

    def __init__(self, layers, alpha=0.1):
        self.W=[]
        self.layers=layers
        self.alpha=alpha

        for i in np.arange(0, len(layers) - 2):

            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))


        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))


    def __repr__(self):
        return 'Neuralnetwork:{}'.format('-'.join(str(i) for i in self.layers))

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x*(1-x)

    def fit(self,X,Y,epochs=1000,displayupdata=100):
        X=np.c_[X,np.ones((X.shape[0]))]

        for epoch in np.arange(0,epochs):
            for (x,target) in zip(X,Y):
                self.fit_partial(x,target)
            if epoch==0 or (epoch+1)%displayupdata==0:
                loss=self.calculate_loss(X,Y)
                print('[INFO] epoch={},loss={:.7f}'.format(epoch+1,loss))


    def fit_partial(self,x,y):
        # print("x={},y={}".format(x,y))
        A=[np.atleast_2d(x)]
        # print('A={}'.format(A))
        for layer in np.arange(0,len(self.W)):
            net=A[layer].dot(self.W[layer])
            out=self.sigmoid(net)
            A.append(out)
            # print('A={}'.format(A))
        error=A[-1]-y
        # print('error={}'.format(error))
        D=[error*self.sigmoid_deriv(A[-1])]
        # print('D={}'.format(D))
        for layer in np.arange(len(A)-2,0,-1):
            delta=D[-1].dot(self.W[layer].T)
            delta=delta*self.sigmoid_deriv(A[layer])
            D.append(delta)
            # print('D={}'.format(D))
        # print(D)
        D=D[::-1]
        # print(D)
        for layer in np.arange(0,len(self.W)):

            self.W[layer]+=-self.alpha*A[layer].T.dot(D[layer])
        # print('w={}'.format(self.W))

    def predict(self,X,addBias=True):
        p=np.atleast_2d(X)
        if addBias:
            p=np.c_[p,np.ones((p.shape[0]))]
        for layer in np.arange(0,len(self.W)):
            p=self.sigmoid(p.dot(self.W[layer]))
        return p

    def calculate_loss(self,X,targets):
        targets=np.atleast_2d(targets)
        predictions=self.predict(X,addBias=False)
        loss=0.5*np.sum((predictions-targets)**2)
        return loss











