mkdir snpx/datasets
mkdir snpx/datasets/CIFAR-10
cd snpx/datasets/CIFAR-10
# wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
cp cifar-10-batches-py/* .
cd ../../../
python3 -c "from snpx.snpx_tf.tf_dataset import CIFAR10;import numpy as np;a = CIFAR10();a.write_to_tfrecord()"