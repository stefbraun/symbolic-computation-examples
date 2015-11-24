import numpy as np
import matplotlib.pyplot as plt
import pickle

# Number of training steps
training_steps = 20000

# Load data
with open('../data.pickle') as f:
    X, y, W, b = pickle.load(f)
X = np.float32(X)
W = np.float32(W)
b = np.float32(b)
num_examples = np.float32(X.shape[0])
num_dimensions = X.shape[1]  # dimensionality
num_labels = len(np.unique(y))  # number of classes

# some hyperparameters
step_size = np.float32(1)
reg = np.float32(1e-3)  # regularization strength

# gradient descent loop

for i in xrange(training_steps):

    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = np.float32(0.5) * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    print "iteration %d: loss %f" % (i, loss)

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg * W  # regularization gradient

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    print np.sum(W) + np.sum(b)

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
