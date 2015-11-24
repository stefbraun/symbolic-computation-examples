import numpy as np
import matplotlib.pyplot as plt
import pickle

# Number of training steps
training_steps = 20000

# Load data
with open('../data.pickle') as f:
    X, y, cl1, cl2 = pickle.load(f)
X = np.float32(X)
num_examples = np.float32(X.shape[0])
num_dimensions = X.shape[1]  # dimensionality
num_labels = len(np.unique(y))  # number of classes

# Initialize parameters randomly
h = 100  # size of hidden layer
W1 = np.float32(0.01 * np.ones((num_dimensions, h)))
b1 = np.float32(np.zeros((1, h)))
W2 = np.float32(0.01 * np.eye(h, num_labels))
b2 = np.float32(np.zeros((1, num_labels)))

# Some hyperparameters
step_size = np.float32(1e-0)
reg = np.float32(1e-3)  # regularization strength

# gradient descent loop

for i in xrange(training_steps):

    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = np.float32(0.5) * reg * np.sum(W1 * W1) + np.float32(0.5) * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    #if i % 1000 == 0:
    print "iteration %d: loss %f" % (i, loss)

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    # perform a parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
#print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
print(np.sum(W1)+np.sum(b1))