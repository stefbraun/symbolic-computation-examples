import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

# Parameters
reg_inp = 1e-3
training_steps = 20000

# Start TensorFlow session
sess = tf.InteractiveSession()

with open('data.pickle') as f:
    X_inp, y_cat, W_inp, b_inp = pickle.load(f)

#Get sample and label count
num_examples = X_inp.shape[0]
num_labels = len(np.unique(y_cat))

# Convert to float32
X_inp = np.float32(X_inp)
W_inp = np.float32(W_inp)
b_inp = np.float32(b_inp)

# Convert label vector to one hot variant
y_inp = np.eye(3)[y_cat]

# Declare Tensorflow symbolic variables
X = tf.placeholder('float', shape=[None, 2])
y_ = tf.placeholder('float', shape=[None, 3])
W = tf.Variable(W_inp)  # 0.01 * np.random.randn(D, K)
b = tf.Variable(b_inp)  # np.zeros((1, K))
reg = tf.constant(reg_inp)

# Initialize variables, i.e. assign initial values '*_inp' to variables
sess.run(tf.initialize_all_variables())

# Softmax function
p_y_given_x = tf.nn.softmax(tf.matmul(X, W) + b)  # same as probs --> simple softmax!

# Loss functions
corect_logprobs = -tf.reduce_sum(y_*tf.log(p_y_given_x))

data_loss = tf.reduce_sum(corect_logprobs) / num_examples

reg_loss = 0.5 * reg * tf.reduce_sum(W * W)

loss = data_loss + reg_loss

# create optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

for i in range(training_steps):
    result = sess.run(loss, feed_dict={X: X_inp, y_: y_inp})
    print "iteration %d: loss %f" % (i, result)
    train_step.run(feed_dict={X: X_inp, y_: y_inp})
    print(np.sum(W.eval())+np.sum(b.eval()))

