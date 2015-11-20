import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import pickle

prec = 'float32'
with open('../data.pickle') as f:
    X_inp, y_inp, W_inp, b_inp = pickle.load(f)
X_inp=X_inp.astype(prec)
W_inp=W_inp.astype(prec)
b_inp=b_inp.astype(prec)
num_examples = X_inp.shape[0]
reg_inp = 1e-3
training_steps = 20000

# Declare Theano symbolic variables
X = T.matrix('X')
y = T.vector('y', dtype='int64')
W = theano.shared(W_inp, name='W')  # 0.01 * np.random.randn(D, K)
b = theano.shared(b_inp, name='b', broadcastable=(True, False))  # np.zeros((1, K))
reg = T.scalar('reg')

print('Initial model:')
print 'W', W.get_value()
print 'b', b.get_value()

for i in xrange(training_steps):

    # Scores function
    scores = T.dot(X, W) + b
    scores_ver = np.dot(X_inp, W_inp) + b_inp
    exp_scores = T.exp(scores)
    exp_scores_ver = np.exp(scores_ver)
    scores_fun = theano.function([X], exp_scores)

    if np.allclose(scores_fun(X_inp), exp_scores_ver) == True:
        print 'Correct: scores function.'
    else:
        print 'Incorrect: scores function '

    # Softmax
    probs = exp_scores / T.sum(exp_scores, axis=1, keepdims=True)
    probs_ver = exp_scores_ver / np.sum(exp_scores_ver, axis=1, keepdims=True)
    probs_fun = theano.function([X], probs)
    p_y_given_x = T.nnet.softmax(T.dot(X, W) + b)  # same as probs --> simple softmax!

    if np.allclose(probs_fun(X_inp), probs_ver) == True:
        print 'Correct: probs function.'
    else:
        print 'Incorrect: probs function '

    # Loss functions
    corect_logprobs = T.nnet.categorical_crossentropy(p_y_given_x, y)
    corect_logprobs_ver = -np.log(probs_ver[range(num_examples), y_inp])

    data_loss = T.sum(corect_logprobs) / num_examples
    data_loss_ver = np.sum(corect_logprobs_ver) / num_examples

    reg_loss = 0.5 * reg * T.sum(W * W)
    reg_loss_ver = 0.5 * reg_inp * np.sum(W_inp * W_inp)

    loss = data_loss + reg_loss
    loss_ver = data_loss_ver + reg_loss_ver

    loss_fun = theano.function([X, y, reg], loss)

    if np.allclose(loss_fun(X_inp, y_inp, reg_inp), loss_ver) == True:
        print 'Correct: loss function.'
    else:
        print 'Incorrect: loss function.'

    # gradient computation
    gw, gb = T.grad(loss, [W, b])
    dscores_ver = probs_ver
    dscores_ver[range(num_examples), y_inp] -= 1
    dscores_ver /= num_examples
    dW_ver = np.dot(X_inp.T, dscores_ver)
    db_ver = np.sum(dscores_ver, axis=0, keepdims=True)
    dW_ver += reg_inp * W_inp

    W_inp += -dW_ver
    b_inp += -db_ver

    intermediate = theano.function([X, y, reg], [gw, gb])
    gw, gb = intermediate(X_inp, y_inp, reg_inp)
    print 'gw', np.allclose(gw, dW_ver)
    print 'gb', np.allclose(gb, db_ver)

    # Compile
    train = theano.function(inputs=[X, y, reg], outputs=[loss], updates=((W, W - gw), (b, b - gb)))

    err = train(X_inp, y_inp, reg_inp)
    print 'iteration %d', i
    print err[0]
    print np.sum(W_inp) + np.sum(b_inp)

