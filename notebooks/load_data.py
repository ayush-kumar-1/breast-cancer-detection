import tensorflow as tf 
import torch
import numpy as np
import os 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from PIL import Image

def load_train_data(): 
    training_data = ["../data/archive/training10_0/training10_0.tfrecords", 
        "../data/archive/training10_1/training10_1.tfrecords",
        "../data/archive/training10_2/training10_2.tfrecords",
        "../data/archive/training10_3/training10_3.tfrecords",
        "../data/archive/training10_4/training10_4.tfrecords"]

    images=[]
    labels=[]
    feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }

    def _parse_function(example, feature_dictionary=feature_dictionary):
        parsed_example = tf.io.parse_example(example, feature_dictionary)
        return parsed_example

    def read_data(filename):
        full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        full_dataset = full_dataset.shuffle(buffer_size=31000)
        full_dataset = full_dataset.cache()
        print("Size of Training Dataset: ", len(list(full_dataset)))
        
        feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }   

        full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(full_dataset)
        for image_features in full_dataset:
            image = image_features['image'].numpy()
            image = tf.io.decode_raw(image_features['image'], tf.uint8)
            image = tf.reshape(image, [299, 299])        
            image=image.numpy()
            #plt.imshow(image)
            images.append(image)
            labels.append(image_features['label_normal'].numpy())

    for file in training_data:
        read_data(file)

    return images, labels

def load_test_data():
    # Load .npy file
    test_data = np.load('../data/archive/test10_data/test10_data.npy')
    test_labels = np.load('../data/archive/test10_labels.npy')

    cv_data = np.load('../data/archive/cv10_data/cv10_data.npy')
    cv_labels = np.load('../data/archive/cv10_labels.npy')

    # combine test and cv into single test set
    test_data = np.concatenate((test_data, cv_data), axis=0)
    test_labels = np.concatenate((test_labels, cv_labels), axis=0)

    return test_data, test_labels