import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from softmax_logic import softmaxregression
from data_load import load_language_data
from features.lang_pattern import sentence_features
import numpy as np

model = softmaxregression()
data,labels = load_language_data(data_dir="data")

X = np.array([sentence_features(s) for s in data])
y = np.array(labels).flatten()

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

model.fit(X_train, y_train, lr=0.1, epochs=1000)

train_preds, _ = model.predict(X_train)
test_preds, _ = model.predict(X_test)

print("Train accuracy:", np.mean(train_preds == y_train))
print("Test accuracy:", np.mean(test_preds == y_test))

from KNN import knn

modelk = knn(k=3)
modelk.fit(X_train, y_train)

knn_train_preds = modelk.predict(X_train)
knn_test_preds = modelk.predict(X_test)

print("k-NN Train accuracy:", np.mean(knn_train_preds == y_train))
print("k-NN Test accuracy:", np.mean(knn_test_preds == y_test))

hard_test =['je vais a la biblioteque',
'ich LIEBE schokolade!!!',
'this is soooo good',
'merci bcp pour ton aide',
'DER zug KOMMT spaet']

hard_labels = [1,2,0,1,2]

X_hard = np.array([sentence_features(s) for s in hard_test])
X_hard = (X_hard - mean) / std  

preds, _ = model.predict(X_hard)
print("Hard test accuracy:", np.mean(preds == hard_labels))

















