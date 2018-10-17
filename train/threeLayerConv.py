######parameter
save_path='threeLayerConv'
layer1Output=1024
layer2Output=512
layer3Output=256
layer4Output=64
#########

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
startTime=time.time()
# number 1 to 10 data
#下面這個配合readmodel
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

    tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def readModel(imgInput):
    result=[]
    with tf.Session() as sess:
        #First let's load meta graph and restore weights 
        saver = tf.train.import_meta_graph('/win/code/python/deeplearn/project/model/{}/{}.meta'.format(save_path,save_path))
        saver.restore(sess,tf.train.latest_checkpoint('/win/code/python/deeplearn/project/model/{}/'.format(save_path)))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data


        graph = tf.get_default_graph()
        #########要換的時候要改下面這裡(tensor name 不一定一樣
        xs = graph.get_tensor_by_name("xs:0")
        #########要換的時候要改下面這裡
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        feed_dict ={xs:imgInput,keep_prob:1}

        #Now, access the op that you want to run. 
        #########要換的時候要改下面這裡
        op_to_restore = graph.get_tensor_by_name("prediction:0")
        predict=sess.run(op_to_restore,feed_dict)
        for item in predict:
            argmax=np.argmax(item)
            print(argmax)
            result.append(argmax)

    return result 

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def normalization(data_input,excu=False):
    if excu==True:
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
    else:
        return data_input





if __name__=='__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784],name='xs')
    ys = tf.placeholder(tf.float32, [None, 10],name='ys')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # print(x_image.shape)  # [n_samples, 28,28,1]

    ## conv1 layer ##
    W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
    b_conv1 =tf.Variable(tf.constant(0.1, shape=[32]))
    batchNormalization_conv1=normalization(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(batchNormalization_conv1)
    h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

    ## conv2 layer ##
    W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
    b_conv2 =tf.Variable(tf.constant(0.1, shape=[64]))
    batchNormalization_conv2=normalization(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.relu(batchNormalization_conv2)
    h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

    ## fc1 layer ##
    W_fc1=tf.Variable(tf.truncated_normal([7*7*64,layer1Output], stddev=0.01))
    b_fc1=tf.Variable(tf.constant(0.1, shape=[layer1Output]))
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    batchNormalization_fc1=normalization(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
    h_fc1= tf.nn.relu(batchNormalization_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2=tf.Variable(tf.truncated_normal([layer1Output,layer2Output], stddev=0.01))
    b_fc2=tf.Variable(tf.constant(0.1, shape=[layer2Output]))
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    batchNormalization_fc2=normalization(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2= tf.nn.relu(batchNormalization_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    ## fc3 layer ##
    '''
    W_fc3=tf.Variable(tf.truncated_normal([layer2Output,layer3Output], stddev=0.01))
    b_fc3=tf.Variable(tf.constant(0.1, shape=[layer3Output]))
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    '''
    '''

    ## fc4 layer ##
    W_fc4=tf.Variable(tf.truncated_normal([layer3Output,layer4Output], stddev=0.01))
    b_fc4=tf.Variable(tf.constant(0.1, shape=[layer4Output]))
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
    '''

    ##############################33333

    '''
    W_fc3=tf.Variable(tf.truncated_normal([layer2Output,10], stddev=0.01))
    b_fc3=tf.Variable(tf.constant(0.1, shape=[10]))
    '''
    W_fc3=tf.Variable(tf.truncated_normal([layer2Output,10], stddev=0.01))
    b_fc3=tf.Variable(tf.constant(0.1, shape=[10]))


    #預測結果
    prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3,name='prediction')


    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss

    #優化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    ##################session

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(save_relative_paths=False)

    for i in range(3001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 200== 0:
            print(compute_accuracy(
                mnist.test.images[:500], mnist.test.labels[:500]))

    saver.save(sess,'/win/code/python/deeplearn/project/model/{}/{}'.format(save_path,save_path))

    finishTime=time.time()
    print('cost time',finishTime-startTime)


