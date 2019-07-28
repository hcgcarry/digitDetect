
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
save_path="dnn6and8"
def readModel(imgInput):
    result=[]
    with tf.Session() as sess:
        #First let"s load meta graph and restore weights 
        #########要換的時候要改下面這裡
        saver = tf.train.import_meta_graph("/win/code/python/deeplearn/project/model/{}/{}.meta".format(save_path,save_path))
        #########要換的時候要改下面這裡
        saver.restore(sess,tf.train.latest_checkpoint("/win/code/python/deeplearn/project/model/{}/".format(save_path)))

        # Now, let"s access and create placeholders variables and
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
            if np.argmax(item)==np.argmax([0,1]):
                print(6)
                result.append(6)
            elif np.argmax(item)==np.argmax([1,0]):
                print(8)
                result.append(8)



    return result


def add_layer(inputs, Weights,biases , scale,shift,activation_function=None, norm=False,keep_prob=1):

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
        #scale = tf.Variable(tf.ones([out_size]))
        #shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema= tf.train.ExponentialMovingAverage(decay=0.5)
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
    global prediction#python要使用再函數裏面使用痊癒變數就要加這行
    y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
#argmax 第二個參數1式按column的意思
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),reduction_indices=0)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_prob:1})
    return result

if __name__=="__main__":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784],name="xs") # 28x28
    ys = tf.placeholder(tf.float32, [None, 2],name="ys")
    keep_prob=tf.placeholder(tf.float32,name="keep_prob")

    ############################network parameter
    normal=True
    activateFunc=tf.nn.leaky_relu
    activateFunc=tf.nn.relu
    #activateFunc=tf.nn.sigmoid
    l1in_size=784
    l1out_size=512
    l2in_size=l1out_size
    l2out_size=256
    finalInputSize=l2out_size

    ############################layer1 
    l1Weights = tf.Variable(tf.random_normal([l1in_size, l1out_size], mean=0., stddev=0.01))
    l1biases = tf.Variable(tf.zeros([1, l1out_size]) + 0.2)
    l1scale = tf.Variable(tf.ones([l1out_size]))
    l1shift = tf.Variable(tf.zeros([l1out_size]))
    ###################################layer2
    l2Weights = tf.Variable(tf.random_normal([l2in_size, l2out_size], mean=0., stddev=0.01))
    l2biases = tf.Variable(tf.zeros([1, l2out_size]) + 0.2)
    l2scale = tf.Variable(tf.ones([l2out_size]))
    l2shift = tf.Variable(tf.zeros([l2out_size]))
    ###################################call add_layer to build network

    layer1=add_layer(normaliztion(xs), l1Weights, l1biases,l1scale,l1shift,  activation_function=activateFunc,norm=normal,keep_prob=keep_prob)
    layer2=add_layer(layer1, l2Weights,l2biases,l2scale,l2shift,  activation_function=activateFunc,norm=normal,keep_prob=keep_prob)
    #prediction = add_layer(layer2, finalInputSize, 10,  activation_function=tf.nn.softmax)#注意最後依從輸出不佳norm
    W_fc2=tf.Variable(tf.truncated_normal([finalInputSize,2], stddev=0.1))
    b_fc2=tf.Variable(tf.constant(0.1, shape=[1,2]))
    prediction = tf.nn.softmax(tf.matmul(layer2, W_fc2) + b_fc2,name="prediction")

    ###############################prediction and gradientdescent 
    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                          reduction_indices=[1]))       # loss


    #train_step= tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
    train_step= tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    #train_step= tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #####################################session
    sess = tf.Session()
    saver=tf.train.Saver()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    init = tf.global_variables_initializer()
    sess.run(init)
    trainAccuracyList=[]
    testAccuracyList=[]
    step=[]

    for i in range(1000):
        batch_ys=[]
        batch_xs=[]
        originBatch_xs, originBatch_ys = mnist.train.next_batch(500)
        for index,item in enumerate(originBatch_ys):
            if np.argmax(item)==7:
                batch_xs.append(originBatch_xs[index])
                batch_ys.append([1,0])
            elif np.argmax(item)==5:
                batch_xs.append(originBatch_xs[index])
                batch_ys.append([0,1])
        batch_ys=np.array(batch_ys)
        batch_xs=np.array(batch_xs)

        #train
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:0.5})
        """
        if i % 100== 0:
            #test accuracy
            print(i,"step")
            trainAccuracy=compute_accuracy(batch_xs,batch_ys)
            testAccuracy=compute_accuracy(mnist.test.images,mnist.test.labels)
            trainAccuracyList.append(trainAccuracy)
            testAccuracyList.append(testAccuracy)
            step.append(i)

            print("train accuracy:",trainAccuracy)
            print("test accuracy",testAccuracy)
            print("###############################")
        """

    save_path = saver.save(sess,"/win/code/python/deeplearn/project/model/{}/{}".format(save_path,save_path))
    plt.figure("accuracy")
    plt.plot(step,trainAccuracyList,c="b",label="trainAccuracy")
    plt.plot(step,testAccuracyList,c="r",label="testAccuracy")
    plt.legend(loc="lower right")
    plt.show()
