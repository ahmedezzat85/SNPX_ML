DATASETS = {'CIFAR-10': {'type': 'image_classification', 'num_classes': 10, 'shape': (32,32,3), 
                         'train_file': 'CIFAR-10_train.tfrecords',
                         'val_file': 'CIFAR-10_val.tfrecords'}
        }

z = DATASETS['CIFAR-10']
print (z['type'])