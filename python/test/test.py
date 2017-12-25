import numpy as np
import snpx
from snpx.snpx_tf.tf_dataset import CIFAR10

d = CIFAR10()
d.write_to_tfrecord()

# mean = np.mean(x_train, axis=0)
# print (mean.shape)
# mean.tofile('cifar10.mean')

# b = np.fromfile('D:\github_ahmedezzat85\SNPX_ML\python\snpx\datasets\CIFAR-10\CIFAR-10.mean')
# print (b.shape)
# print (b)