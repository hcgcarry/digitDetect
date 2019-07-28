import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# the kernel can be "linear", "poly" or "rbf"
# the probability was set to True so as to show
# how sure the model is of it"s prediction
batch_xs=mnist.train.images[:60000]
batch_ys=mnist.train.labels[:60000]
svc= SVC(kernel="linear", probability=True)

# let"s train the model with all the input data
svc.fit(batch_xs,batch_ys)

print(model.predict(mnist.test.images[:4, :]))
print(mnist.test.labels[:4])
# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don"t need to train the model again
save_name="svc"
save_directory ="/win/code/python/deeplearn/project/model/svc/svc.pkl"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model,save_directory)



