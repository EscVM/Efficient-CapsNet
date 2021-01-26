# Copyright 2021 Adam Byerly & Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
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
import cv2
tf2 = tf.compat.v2

# constants
MNIST_IMG_SIZE = 28
MNIST_TRAIN_IMAGE_COUNT = 60000
PARALLEL_INPUT_CALLS = 16

# normalize dataset
def pre_process(image, label):
    return (image / 256)[...,None].astype('float32'), tf.keras.utils.to_categorical(label, num_classes=10)

def image_shift_rand(image, label):
    image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
    nonzero_x_cols = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=0), 0)), tf.int32)
    nonzero_y_rows = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=1), 0)), tf.int32)
    left_margin = tf.reduce_min(nonzero_x_cols)
    right_margin = MNIST_IMG_SIZE - tf.reduce_max(nonzero_x_cols) - 1
    top_margin = tf.reduce_min(nonzero_y_rows)
    bot_margin = MNIST_IMG_SIZE - tf.reduce_max(nonzero_y_rows) - 1
    rand_dirs = tf.random.uniform([2])
    dir_idxs = tf.cast(tf.floor(rand_dirs * 2), tf.int32)
    rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
    x_amts = [tf.floor(-1.0 * rand_amts[0] *
              tf.cast(left_margin, tf.float32)), tf.floor(rand_amts[0] *
              tf.cast(1 + right_margin, tf.float32))]
    y_amts = [tf.floor(-1.0 * rand_amts[1] *
              tf.cast(top_margin, tf.float32)), tf.floor(rand_amts[1] *
              tf.cast(1 + bot_margin, tf.float32))]
    x_amt = tf.cast(tf.gather(x_amts, dir_idxs[1], axis=0), tf.int32)
    y_amt = tf.cast(tf.gather(y_amts, dir_idxs[0], axis=0), tf.int32)
    image = tf.reshape(image, [MNIST_IMG_SIZE * MNIST_IMG_SIZE])
    image = tf.roll(image, y_amt * MNIST_IMG_SIZE, axis=0)
    image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
    image = tf.transpose(image)
    image = tf.reshape(image, [MNIST_IMG_SIZE * MNIST_IMG_SIZE])
    image = tf.roll(image, x_amt * MNIST_IMG_SIZE, axis=0)
    image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
    image = tf.transpose(image)
    image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1])
    return image, label

def image_rotate_random_py_func(image, angle):
    rot_mat = cv2.getRotationMatrix2D(
        (MNIST_IMG_SIZE/2, MNIST_IMG_SIZE/2), int(angle), 1.0)
    rotated = cv2.warpAffine(image.numpy(), rot_mat,
        (MNIST_IMG_SIZE, MNIST_IMG_SIZE))
    return rotated

def image_rotate_random(image, label):
    rand_amts = tf.maximum(tf.minimum(
        tf.random.normal([2], 0, .33), .9999), -.9999)
    angle = rand_amts[0] * 30  # degrees
    new_image = tf.py_function(image_rotate_random_py_func,
        (image, angle), tf.float32)
    new_image = tf.cond(rand_amts[1] > 0, lambda: image, lambda: new_image)
    return new_image, label

def image_erase_random(image, label):
    sess = tf.compat.v1.Session()
    with sess.as_default():
        rand_amts = tf.random.uniform([2])
        x = tf.cast(tf.floor(rand_amts[0]*19)+4, tf.int32)
        y = tf.cast(tf.floor(rand_amts[1]*19)+4, tf.int32)
        patch = tf.zeros([4, 4])
        mask = tf.pad(patch, [[x, MNIST_IMG_SIZE-x-4],
            [y, MNIST_IMG_SIZE-y-4]],
            mode='CONSTANT', constant_values=1)
        image = tf.multiply(image, tf.expand_dims(mask, -1))
        return image, label
    
    
def image_squish_random(image, label):
    rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
    width_mod = tf.cast(tf.floor(
        (rand_amts[0] * (MNIST_IMG_SIZE / 4)) + 1), tf.int32)
    offset_mod = tf.cast(tf.floor(rand_amts[1] * 2.0), tf.int32)
    offset = (width_mod // 2) + offset_mod
    image = tf.image.resize(image,
        [MNIST_IMG_SIZE, MNIST_IMG_SIZE - width_mod],
        method=tf2.image.ResizeMethod.LANCZOS3,
        preserve_aspect_ratio=False,
        antialias=True)
    image = tf.image.pad_to_bounding_box(
        image, 0, offset, MNIST_IMG_SIZE, MNIST_IMG_SIZE + offset_mod)
    image = tf.image.crop_to_bounding_box(
        image, 0, 0, MNIST_IMG_SIZE, MNIST_IMG_SIZE)
    return image, label

def generator(image, label):
    return (image, label), (label, image)

def generate_tf_data(X_train, y_train, X_test, y_test, batch_size):
	dataset_train = tf.data.Dataset.from_tensor_slices((X_train,y_train))
	dataset_train = dataset_train.shuffle(buffer_size=MNIST_TRAIN_IMAGE_COUNT)
	dataset_train = dataset_train.map(image_rotate_random)
	dataset_train = dataset_train.map(image_shift_rand,
	    num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.map(image_squish_random,
	    num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.map(image_erase_random,
	   num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.map(generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.batch(batch_size)
	dataset_train = dataset_train.prefetch(-1)

	dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
	dataset_test = dataset_test.cache()
	dataset_test = dataset_test.map(generator,
	    num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_test = dataset_test.batch(batch_size)
	dataset_test = dataset_test.prefetch(-1)
    
	return dataset_train, dataset_test