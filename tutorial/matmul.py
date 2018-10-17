import tensorflow as tf
a=tf.constant([2,2,2,1,1,1],shape=[2,3])
b=tf.constant([1,3,4],shape=[3,1])
result=tf.matmul(b,a)
with tf.Session() as sess:
    print(sess.run(result))

