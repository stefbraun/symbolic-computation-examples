import numpy as np
import matplotlib.pyplot as plt
import pickle

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X_inp = np.zeros((N*K,D)) # data matrix (each row = single example)
y_inp = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X_inp[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y_inp[ix] = j
# lets visualize the data:
plt.scatter(X_inp[:, 0], X_inp[:, 1], c=y_inp, s=40, cmap=plt.cm.Spectral)

W_inp = 0.01 * np.random.randn(D, K)
b_inp = np.zeros((1, K))

with open('data.pickle', 'w') as f:
    pickle.dump([X_inp, y_inp, W_inp, b_inp], f)

