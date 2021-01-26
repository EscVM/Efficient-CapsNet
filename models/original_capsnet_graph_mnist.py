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
from utils.layers_hinton import PrimaryCaps, DigitCaps, Length, Mask
import tensorflow_addons as tfa


def capsnet_graph(input_shape, routing):
    """
    Original CapsNet graph architecture described in "dynamic routinig between capsules".
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    routing: int
        number of routing iterations
    """
    inputs = tf.keras.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(256, 9, activation="relu")(inputs)
    primary = PrimaryCaps(C=32, L=8, k=9, s=2)(x)
    digit_caps = DigitCaps(10, 16, routing=routing)(primary)  
    digit_caps_len = Length(name='capsnet_output_len')(digit_caps)
    pr_shape = primary.shape
    primary = tf.reshape(primary,(-1,pr_shape[1]*pr_shape[2]*pr_shape[3],pr_shape[-1]))

    return tf.keras.Model(inputs=inputs,outputs=[primary, digit_caps, digit_caps_len] , name='Original_CapsNet')


def generator_graph(input_shape):
    """
    Generator graph architecture.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16*10)
    
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, n_routing, verbose):
    """
    Original CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    mode: str
        working mode ('train' & 'test')
    n_routing: int
        number of routing iterations
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.Input(shape=(10))
    noise = tf.keras.layers.Input(shape=(10, 16))
    
    capsnet = capsnet_graph(input_shape, routing=n_routing)
    primary, digit_caps, digit_caps_len = capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise]) # only if mode is play
    
    if verbose:
        capsnet.summary()
        print("\n\n")
    

    masked_by_y = Mask()([digit_caps, y_true])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digit_caps)  # Mask using the capsule with maximal length. For prediction
    masked_noised_y = Mask()([noised_digitcaps, y_true])
    

    generator = generator_graph(input_shape)
    
    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)
      

    if mode == 'train':   
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
