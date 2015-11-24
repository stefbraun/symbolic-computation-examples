import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

# Number of training steps
training_steps = 20000

# Start TensorFlow session
sess = tf.InteractiveSession()

# Load data
with open('../data.pickle') as f:
   X_inp, y_cat, W_inp, b_inp = pickle.load(f)
X_inp = np.float32(X_inp)
num_examples = X_inp.shape[0]
num_dimensions = X_inp.shape[1]
num_labels = len(np.unique(y_cat))

# Initialize parameters randomly
h = 100  # size of hidden layer
W1_inp = np.float32(0.01 * np.ones((num_dimensions, h)))
b1_inp = np.float32(np.zeros((1, h)))
W2_inp = np.float32(0.01 * np.eye(h, num_labels))
b2_inp = np.float32(np.zeros((1, num_labels)))

# Some hyperparameters
step_size = np.float32(1e-0)
reg_inp = np.float32(1e-3)  # regularization strength

# Convert label vector to one hot variant
y_inp = np.eye(3)[y_cat]

# Declare TensorFlow symbolic variables
X = tf.placeholder('float', shape=[None, 2])
y_ = tf.placeholder('float', shape=[None, 3])
W1 = tf.Variable(W1_inp)  # 0.01 * np.random.randn(D, K)
b1 = tf.Variable(b1_inp)  # np.zeros((1, K))
W2 = tf.Variable(W2_inp)
b2 = tf.Variable(b2_inp)
reg = tf.constant(reg_inp)

# Initialize variables, i.e. assign initial values '*_inp' to variables
sess.run(tf.initialize_all_variables())

# Evaluate class scores
hidden_layer = tf.maximum(0.0, tf.matmul(X, W1)+b1)
scores = tf.matmul(hidden_layer, W2)+b2

# compute the loss: average cross-entropy loss and regularization
p_y_given_x = tf.nn.softmax(scores)

corect_logprobs = -tf.reduce_sum(y_*tf.log(p_y_given_x))

data_loss = tf.reduce_sum(corect_logprobs) / num_examples

reg_loss = 0.5 * reg * tf.reduce_sum(W1 * W1) + 0.5 * reg * tf.reduce_sum(W2*W2)

loss = data_loss + reg_loss

# create optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

# gradient descent loop
for i in range(training_steps):
    result = sess.run(loss, feed_dict={X: X_inp, y_: y_inp})
    print "iteration %d: loss %f" % (i, result)
    train_step.run(feed_dict={X: X_inp, y_: y_inp})
    print(np.sum(W1.eval())+np.sum(b1.eval()))

5+5