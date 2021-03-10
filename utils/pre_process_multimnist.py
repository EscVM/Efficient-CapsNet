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
import cv2

# constants
MULTIMNIST_IMG_SIZE = 36

def pad_dataset(images,pad):
    return np.pad(images,[(0,0),(pad,pad),(pad,pad)])

def pre_process(image, label):
    return (image / 255)[...,None].astype('float32'), tf.keras.utils.to_categorical(label, num_classes=10)

def shift_images(images, shifts, max_shift):
    l = images.shape[1]
    images_sh = np.pad(images,((0,0),(max_shift,max_shift),(max_shift,max_shift),(0,0)))
    shifts = max_shift - shifts
    batches = np.arange(len(images))[:,None,None]
    images_sh = images_sh[batches,np.arange(l+max_shift*2)[None,:,None],(shifts[:,0,None]+np.arange(0,l))[:,None,:]]
    images_sh = images_sh[batches,(shifts[:,1,None]+np.arange(0,l))[...,None],np.arange(l)[None,None]]
    return images_sh

def merge_with_image(images,labels,i,shift,n_multi=1000): #for an image i, generate n_multi merged images
    base_image = images[i]
    base_label = labels[i]
    indexes = np.arange(len(images))[np.bitwise_not((labels==base_label).all(axis=-1))]
    indexes = np.random.choice(indexes,n_multi,replace=False)
    top_images = images[indexes]
    top_labels = labels[indexes]
    shifts = np.random.randint(-shift,shift+1,(n_multi+1,2))
    images_sh = shift_images(np.concatenate((base_image[None],top_images),axis=0),shifts,shift)
    base_sh = images_sh[0]
    top_sh = images_sh[1:]
    merged = np.clip(base_sh+top_sh,0,1)
    merged_labels = base_label+top_labels
    return merged,merged_labels

def multi_mnist_generator(images,labels,shift):
    def multi_mnist():
        while True:
            i = np.random.randint(len(images))
            j = np.random.randint(len(images))
            while np.all(images[i]==images[j]):
                j = np.random.randint(len(images))
            base = shift_images(images[i:i+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
            top = shift_images(images[j:j+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
            merged = tf.clip_by_value(tf.add(base, top),0,1)
            yield (merged,labels[i],labels[j]),(labels[i]+labels[j],base,top)
    return multi_mnist
    
def multi_mnist_generator_validation(images,labels,shift):
    def multi_mnist_val():
        for i in range(len(images)):
            j = np.random.randint(len(images))
            while np.all(labels[i]==labels[j]):
                j = np.random.randint(len(images))
            base = shift_images(images[i:i+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
            top = shift_images(images[j:j+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
            merged = tf.clip_by_value(tf.add(base, top),0,1)
            yield (merged,labels[i],labels[j]),(labels[i]+labels[j],base,top)
    return multi_mnist_val

def multi_mnist_generator_test(images,labels,shift,n_multi=1000):
    def multi_mnist_test():
        for i in range(len(images)):
            X_merged,y_merged = merge_with_image(images,labels,i,shift,n_multi)
            yield X_merged,y_merged
    return multi_mnist_test 

def generate_tf_data(X_train, y_train, X_test, y_test, batch_size, shift):
    input_shape = (MULTIMNIST_IMG_SIZE,MULTIMNIST_IMG_SIZE,1)
    dataset_train = tf.data.Dataset.from_generator(multi_mnist_generator(X_train,y_train,shift),
                                                   output_shapes=((input_shape,(10,),(10,)),((10,),input_shape,input_shape)),
                                                   output_types=((tf.float32,tf.float32,tf.float32),
                                                                 (tf.float32,tf.float32,tf.float32)))
    dataset_train = dataset_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset_test = tf.data.Dataset.from_generator(multi_mnist_generator_validation(X_test,y_test,shift),
                                                 output_shapes=((input_shape,(10,),(10,)),((10,),input_shape,input_shape)),
                                                 output_types=((tf.float32,tf.float32,tf.float32),
                                                               (tf.float32,tf.float32,tf.float32)))
    dataset_test = dataset_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset_train, dataset_test

def generate_tf_data_test(X_test, y_test, shift, n_multi=1000, random_seed=42):
    input_shape = (MULTIMNIST_IMG_SIZE,MULTIMNIST_IMG_SIZE,1)
    np.random.seed(random_seed)
    dataset_test = tf.data.Dataset.from_generator(multi_mnist_generator_test(X_test,y_test,shift,n_multi),
                                                  output_shapes=((n_multi,)+input_shape,(n_multi,10,)),
                                                  output_types=(tf.float32,tf.float32))
    dataset_test = dataset_test.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset_test
