import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


 
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
 
 
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
 
 
w_h = init_weights([784, 1024])
w_h2 = init_weights([1024, 625])
w_o = init_weights([625, 10])
 
 
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
 
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
 
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
 
    return tf.matmul(h2, w_o)
 
 
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1)), tf.float32))
epoch_count = 1000
batch_size = 50
keep_input = 0.8
keep_hidden = 0.75
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    for i in range(epoch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, p_keep_input: keep_input, p_keep_hidden: keep_hidden})
        if i % 100 == 0:
            loss, acc = sess.run([cost, predict_acc], feed_dict={X: batch_x, Y: batch_y, p_keep_input: 1., p_keep_hidden: 1.})
            print("Epoch: {}".format(step), "\tLoss: {:.6f}".format(loss), "\tTraining Accuracy: {:.5f}".format(acc))
    print("Testing Accuracy: {:0.5f}".format(sess.run(predict_acc, feed_dict={X: mnist.test.images, Y: mnist.test.labels, p_keep_input: 1., p_keep_hidden: 1.})))
