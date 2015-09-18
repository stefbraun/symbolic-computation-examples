import numpy as np
import matplotlib.pyplot as plt
import pickle
import theano.tensor as T
import theano

with open('data.pickle') as f:
    X_inp, y_inp, cl1, cl2 = pickle.load(f)

D = 2  # dimensionality
K = 3  # number of classes

# initialize parameters randomly
h = 100  # size of hidden layer
W1_inp = 0.01 * np.ones((D, h))
b1_inp = np.zeros((1, h))
W2_inp = 0.01 * np.eye(h, K)
b2_inp = np.zeros((1, K))

# some hyperparameters
step_size = 1e-0
reg_inp = 1e-3  # regularization strength

# Declare Theano symbolic variables
X = T.matrix(name='X')
y = T.vector(name='y', dtype='int64')
W1 = theano.shared(W1_inp, name='W1')
b1 = theano.shared(b1_inp, name='b1', broadcastable=(True, False))
W2 = theano.shared(W2_inp, name='W2')
b2 = theano.shared(b2_inp, name='b2', broadcastable=(True, False))
reg = T.scalar('reg')

# gradient descent loop
num_examples = X_inp.shape[0]

for i in xrange(1000):

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

    loss_val = train(X_inp, y_inp, reg_inp)
    if i % 10 == 0:
        print "iteration %d: loss_sym %f" % (i, loss_val[0])
