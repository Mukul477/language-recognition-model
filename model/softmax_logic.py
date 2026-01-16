import numpy as np

class softmaxregression:
    def __init__(self):
        self.w = None
        self.b = None
        self.classes = ["english","french","german"]
    
    def softmax(self,z):
        exp_z = np.exp(z-np.max(z,axis=1,keepdims=True))
        sum_exp_z = np.sum(exp_z,axis=1,keepdims=True)
        return exp_z/sum_exp_z
    
    def fit(self,X,y,lr=0.01,epochs=1000):
        n_samples,n_features = X.shape
        n_classes = np.max(y) + 1
        self.w = np.random.randn(n_features,n_classes)
        self.b = np.zeros(n_classes,)
        for epoch in range(epochs):
            z = X @ self.w + self.b
            softi = self.softmax(z)
            loss = self.loss(y, softi)
            error = softi
            error[np.arange(X.shape[0]), y] -= 1
            dw = X.T @ error / X.shape[0]
            db = np.mean(error, axis=0)
            self.w -= lr * dw
            self.b -= lr * db
            if epoch % 100 == 0:
                print(epoch, loss)
    
    def predict_proba(self,X):
        z = X @ self.w + self.b
        return self.softmax(z)
    
    def predict(self, X):
        z = X @ self.w + self.b
        probs = self.softmax(z)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return preds, confs
    
    def loss(self, y, probs):
        n = y.shape[0]
        return -np.mean(np.log(probs[np.arange(n), y] + 1e-9))
    
            




  


        
        
        


    



        



       



