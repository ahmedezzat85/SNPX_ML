from snpx.snpx_tf.tf_dataset import CIFAR10
import numpy as np

a = CIFAR10()
a.write_to_tfrecord()
X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = a.get_raw_data()
m = np.mean(X_Train, axis=0)
print (m.shape)