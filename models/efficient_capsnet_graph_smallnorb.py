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
from utils.layers import PrimaryCaps, FCCaps, Length, Mask
import tensorflow_addons as tfa


def efficient_capsnet_graph(input_shape):
    """
    Efficient-CapsNet graph architecture.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(32,7,2,activation=None, padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x =   tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.Conv2D(64,3, activation=None, padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x =   tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.Conv2D(64,3, activation=None, padding='valid', kernel_initializer='he_normal')(x) 
    x = tf.keras.layers.LeakyReLU()(x)
    x =   tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.Conv2D(128,3,2, activation=None, padding='valid', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.LeakyReLU()(x)
    x =   tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(x)

    x = PrimaryCaps(128, 8, 16, 8)(x) # there could be an error
    
    digit_caps = FCCaps(5,16)(x)

    
    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs,outputs=[digit_caps,digit_caps_len], name='Efficient_CapsNet')


def generator_graph(input_shape):
    """
    Generator graph architecture.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16*5)

    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.Reshape(target_shape=(8,8,1))(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid", activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3,3), padding="valid", activation=tf.nn.sigmoid)(x)         
    
    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    mode: str
        working mode ('train' & 'test')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(5,))


    efficient_capsnet = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    digit_caps, digit_caps_len = efficient_capsnet(inputs)

    
    masked_by_y = Mask()([digit_caps, y_true])  
    masked = Mask()(digit_caps)
    
    generator = generator_graph(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)

    if mode == 'train':   
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train])
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval])
    else:
        raise RuntimeError('mode not recognized')
