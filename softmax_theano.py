import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import pickle


with open('data.pickle') as f:
    X_inp, y_inp, W_inp, b_inp = pickle.load(f)

num_examples = X_inp.shape[0]
reg_inp = 1e-3
training_steps = 200

# Declare Theano symbolic variables
X = T.matrix('X')
y = T.vector('y', dtype='int64')
W = theano.shared(W_inp, name='W')  # 0.01 * np.random.randn(D, K)
b = theano.shared(b_inp, name='b', broadcastable=(True, False))  # np.zeros((1, K))
reg = T.scalar('reg')

print('Initial model:')
print 'W', W.get_value()
print 'b', b.get_value()


# Scores function
scores = T.dot(X, W) + b
exp_scores = T.exp(scores)
scores_fun = theano.function([X], exp_scores)

# Softmax
# probs = exp_scores / T.sum(exp_scores, axis=1, keepdims=True)
p_y_given_x = T.nnet.softmax(T.dot(X, W) + b)  # same as probs --> simple softmax!

# Loss functions
corect_logprobs = T.nnet.categorical_crossentropy(p_y_given_x, y)

data_loss = T.sum(corect_logprobs) / num_examples

reg_loss = 0.5 * reg * T.sum(W * W)

loss = data_loss + reg_loss

loss_fun = theano.function([X, y, reg], loss)

# gradient computation
gw, gb = T.grad(loss, [W, b])

# Compile
train = theano.function(inputs=[X, y, reg], outputs=[loss], updates=((W, W - gw), (b, b - gb)))

for i in xrange(training_steps):
    err = train(X_inp, y_inp, reg_inp)
    print 'iteration %d', i
    print err[0]
    print np.sum(W_inp) + np.sum(b_inp)

