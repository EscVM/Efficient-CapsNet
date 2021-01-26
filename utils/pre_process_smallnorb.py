# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import os
from tqdm.notebook import tqdm


# constants
SAMPLES = 24300
INPUT_SHAPE = 96
PATCH_SMALLNORB = 48
N_CLASSES = 5
MAX_DELTA = 2.0
LOWER_CONTRAST = 0.5
UPPER_CONTRAST = 1.5
PARALLEL_INPUT_CALLS = 16


def pre_process(ds):
    X = np.empty((SAMPLES, INPUT_SHAPE, INPUT_SHAPE, 2))
    y = np.empty((SAMPLES,))
        
    for index, d in tqdm(enumerate(ds.batch(1))):
        X[index, :, :, 0:1] = d['image']
        X[index, :, :, 1:2] = d['image2']
        y[index] = d['label_category']
    return X, y


def standardize(x, y):
    x[...,0] = (x[...,0] - x[...,0].mean()) / x[...,0].std()
    x[...,1] = (x[...,1] - x[...,1].mean()) / x[...,1].std()
    return x, tf.one_hot(y, N_CLASSES)

def rescale(x, y, config):
    with tf.device("/cpu:0"):
        x = tf.image.resize(x , [config['scale_smallnorb'], config['scale_smallnorb']])
    return x, y

def test_patches(x, y, config):
    res = (config['scale_smallnorb'] - config['patch_smallnorb']) // 2
    return x[:,res:-res,res:-res,:], y


def generator(image, label):
    return (image, label), (label, image)

def random_patches(x, y):
    return tf.image.random_crop(x, [PATCH_SMALLNORB, PATCH_SMALLNORB, 2]), y

def random_brightness(x, y):
    return tf.image.random_brightness(x, max_delta=MAX_DELTA), y

def random_contrast(x, y):
    return tf.image.random_contrast(x, lower=LOWER_CONTRAST, upper=UPPER_CONTRAST), y


def generate_tf_data(X_train, y_train, X_test_patch, y_test, batch_size):
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # dataset_train = dataset_train.shuffle(buffer_size=SAMPLES) not needed if imported with tfds
    dataset_train = dataset_train.map(random_patches,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_brightness,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_contrast,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(-1)

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test_patch, y_test))
    dataset_test = dataset_test.cache()
    dataset_test = dataset_test.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(1)
    dataset_test = dataset_test.prefetch(-1)
    
    return dataset_train, dataset_test
