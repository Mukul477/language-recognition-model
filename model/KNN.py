import numpy as np 

class knn:
    def __init__(self,k=3):
        self.k = k
        
    def fit (self,X,y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X):
        preds = []
        for x in X:
            distances = []                 
            for x_train in self.X_train:
                distances.append(self.euclidean_distance(x, x_train))

            distances = np.array(distances) 
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]

            preds.append(max(set(k_labels), key=k_labels.count))

        return np.array(preds)
    

              