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
    hidden_layer_ver = np.maximum(0, np.dot(X_inp, W1_inp) + b1_inp)  # note, ReLU

    scores = T.dot(hidden_layer, W2) + b2
    scores_ver = np.dot(hidden_layer_ver, W2_inp) + b2_inp

    # compute the class probabilities
    p_y_given_x = T.nnet.softmax(scores)  # same as probs

    exp_scores_ver = np.exp(scores_ver)
    probs_ver = exp_scores_ver / np.sum(exp_scores_ver, axis=1, keepdims=True)  # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = T.nnet.categorical_crossentropy(p_y_given_x, y)
    corect_logprobs_ver = -np.log(probs_ver[range(num_examples), y_inp])

    data_loss = T.sum(corect_logprobs) / num_examples
    data_loss_ver = np.sum(corect_logprobs_ver) / num_examples

    reg_loss = 0.5 * reg * T.sum(W1 * W1) + 0.5 * reg * T.sum(W2 * W2)
    reg_loss_ver = 0.5 * reg_inp * np.sum(W1_inp * W1_inp) + 0.5 * reg_inp * np.sum(W2_inp * W2_inp)

    loss = data_loss + reg_loss
    loss_ver = data_loss_ver + reg_loss_ver

    gw1, gb1, gw2, gb2 = T.grad(loss, [W1, b1, W2, b2])

    # Compile
    train = theano.function(inputs=[X, y, reg], outputs=[loss],
                            updates=[(W1, W1 - gw1), (b1, b1 - gb1), (W2, W2 - gw2), (b2, b2 - gb2)])

    loss_val = train(X_inp, y_inp, reg_inp)
    if i % 10 == 0:
        print "iteration %d: loss_sym %f" % (i, loss_val[0])
        print "iteration %d: loss_pyt %f" % (i, loss_ver)

    # compute the gradient on scores
    dscores = probs_ver
    dscores[range(num_examples), y_inp] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer_ver.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2_inp.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer_ver <= 0] = 0
    # finally into W,b
    dW1 = np.dot(X_inp.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg_inp * W2_inp
    dW1 += reg_inp * W1_inp

    # perform a parameter update
    W1_inp += -step_size * dW1
    b1_inp += -step_size * db1
    W2_inp += -step_size * dW2
    b2_inp += -step_size * db2


# evaluate training set accuracy
hidden_layer_ver = np.maximum(0, np.dot(X_inp, W1_inp) + b1_inp)
scores_ver = np.dot(hidden_layer_ver, W2_inp) + b2_inp
predicted_class = np.argmax(scores_ver, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y_inp))
