import tensorflow as tf
import numpy as np
import pandas as pd
import random as rd
import mnist_data as mn


TRAINING_SET = 'data/train.csv'
TEST_SET = 'data/test.csv'


# read
mm = mn.mnist()
mm.read_csv_train('data/train.csv')
mm.read_csv_test('data/test.csv')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

# Initialization for handy use and also symmetry breaking. We use ReLU neuron
# s.
def weight_variable(shape):
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/constant_op.html#truncated_normal
    intial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(intial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# tf flexibility in convolution and pooling operations also handy use.
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# first convolutional layer

# 32 features for each 5x5 patch
W_conv1 = weight_variable([5,5,1,32])
B_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])

# We then convolve x_image with the weight tensor, add the bias,
# apply the ReLU function, and finally max pool.


h_connv1 = tf.nn.relu(conv2d(x_image,W_conv1) + B_conv1)
h_pool1 = max_pool_2x2(h_connv1)

# second convolutional layer
# 64 featrues
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully-Connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout to overcome overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train :
# we use sophisticated ADAM optimizer instead of the Gradient descent
# train 20000 steps, and log the info in every 100 steps
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# saver
saver = tf.train.Saver()
for i in range(20000):
  batch = mm.read_next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
# add keep_prob to control the dropout rate
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#save variables to the current dir
save_path = saver.save(sess, "mnist_expert_model.ckpt")
print "Model saved in file: ", save_path

predict = []
for i in xrange(len(mm.test_values)/100):
    result = y_conv.eval(feed_dict={
    x: mm.test_next(100),keep_prob: 1.0
})
    predict += list(tf.argmax(result,1).eval())
df = pd.DataFrame(range(len(mm.test_values)),columns=['ImageId'])
df = df.add(1)
df['Label'] = predict

df.to_csv('result.csv',index=False,index_label=False)
