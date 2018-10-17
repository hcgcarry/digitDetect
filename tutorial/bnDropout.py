import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None, norm=False,keep_prob=1):
    # weights and biases (bad initialization for this case)
    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.2)

    # fully connected product
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # normalize fully connected product
    if norm:
        # Batch Normalize
        fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0],   # the dimension you wanna normalize, here [0] for batch 這邊靈的話就是以row為單位計算
                        # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        )
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(ema.average(fc_mean)), tf.identity(ema.average(fc_var))
        mean, var = mean_var_with_update()

        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        # similar with this two steps:
        # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + epsilon)
        # Wx_plus_b = Wx_plus_b * scale + shift

    # activation
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    outputs=tf.nn.dropout(outputs,keep_prob)
    return outputs

def normaliztion(data_input):
        fc_mean, fc_var = tf.nn.moments(
            data_input,
            axes=[0],   # the dimension you wanna normalize, here [0] for batch 這邊靈的話就是以row為單位計算
                        # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(ema.average(fc_mean)), tf.identity(ema.average(fc_var))
        mean, var = mean_var_with_update()
        data_input = tf.nn.batch_normalization(data_input, mean, var, shift, scale, epsilon)
        return data_input





def compute_accuracy(v_xs, v_ys):
    #下面這行使用的時候要改一下
    global prediction#python要使用再函數裏面使用痊癒變數就要加這行
    y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
#argmax 第二個參數1式按column的意思
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),reduction_indices=0)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_prob:1})
    return result


