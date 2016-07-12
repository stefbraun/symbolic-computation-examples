import numpy as np
from timeit import default_timer as timer
import pickle
import theano.tensor as T
import theano
import csv

# Number of training steps
training_steps = 20000

# Load data
with open('../data.pickle') as f:
    X_inp, y_cat, cl1, cl2 = pickle.load(f)
X_inp = np.float32(X_inp)
y_inp = y_cat
num_examples = X_inp.shape[0]
num_dimensions = X_inp.shape[1]  # dimensionality
num_labels = len(np.unique(y_cat))  # number of classes

# Initialize parameters randomly
h = 100  # size of hidden layer
W1_inp = np.float32(0.01 * np.ones((num_dimensions, h)))
b1_inp = np.float32(np.zeros((1, h)))
W2_inp = np.float32(0.01 * np.eye(h, num_labels))
b2_inp = np.float32(np.zeros((1, num_labels)))

# Some hyperparameters
step_size = np.float32(1e-0)
reg_inp = np.float32(1e-3)  # regularization strength

# Declare Theano symbolic variables
X = T.matrix(name='X')
y = T.vector(name='y', dtype='int64')
W1 = theano.shared(W1_inp, name='W1')
b1 = theano.shared(b1_inp, name='b1', broadcastable=(True, False))
W2 = theano.shared(W2_inp, name='W2')
b2 = theano.shared(b2_inp, name='b2', broadcastable=(True, False))
reg = T.scalar('reg')

# gradient descent loop


# evaluate class scores, [N x K]
hidden_layer = T.maximum(0, T.dot(X, W1) + b1)
scores = T.dot(hidden_layer, W2) + b2

# compute the class probabilities
p_y_given_x = T.nnet.softmax(scores)  # same as probs

# compute the loss: average cross-entropy loss and regularization
corect_logprobs = T.nnet.categorical_crossentropy(p_y_given_x, y)

data_loss = T.sum(corect_logprobs) / num_examples

reg_loss = 0.5 * reg * T.sum(W1 * W1) + 0.5 * reg * T.sum(W2 * W2)

loss = data_loss + reg_loss

gw1, gb1, gw2, gb2 = T.grad(loss, [W1, b1, W2, b2])

# Compile
train = theano.function(inputs=[X, y, reg], outputs=[loss],
                        updates=[(W1, W1 - gw1), (b1, b1 - gb1), (W2, W2 - gw2), (b2, b2 - gb2)])
bench_list=[]
for i in xrange(training_steps):
    start=timer()
    loss_val = train(X_inp, y_inp, reg_inp)
    end=timer()
    bench_list.append(end-start)
print('Mean[sec] %f Min[sec] %f Max[sec] %f Median[sec] %f Total[sec] %f' % ( np.mean(bench_list), np.min(bench_list), np.max(bench_list), np.median(bench_list), np.sum(bench_list)))

with open('log.csv', 'a') as f:
    c = csv.writer(f)
    c.writerow(
        [np.mean(bench_list), np.min(bench_list), np.max(bench_list), np.median(bench_list), np.sum(bench_list)])
