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


class SquashHinton(tf.keras.layers.Layer):
    """
    Squash activation function presented in 'Dynamic routinig between capsules'.

    ...
    
    Attributes
    ----------
    eps: int
        fuzz factor used in numeric expression
 
    Methods
    -------
    call(s)
        compute the activation from input capsules

    """

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s,axis=-1,keepdims=True)
        return tf.multiply(n**2/(1+n**2)/(n+self.eps), s)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape



class Squash(tf.keras.layers.Layer):
    """
    Squash activation used in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
    
    ...
    
    Attributes
    ----------
    eps: int
        fuzz factor used in numeric expression
    
    Methods
    -------
    call(s)
        compute the activation from input capsules
    """

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s,axis=-1,keepdims=True)
        return (1 - 1/(tf.math.exp(n)+self.eps))*(s/(n+self.eps))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape




class PrimaryCaps(tf.keras.layers.Layer):
    """
    Create a primary capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'. 
    Properties of each capsule s_n are exatracted using a 2D depthwise convolution.
    
    ...
    
    Attributes
    ----------
    F: int
        depthwise conv number of features
    K: int
        depthwise conv kernel dimension
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    s: int
        depthwise conv strides
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, F, K, N, D, s=1, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.F = F
        self.K = K
        self.N = N
        self.D = D
        self.s = s
        
    def build(self, input_shape):    
        self.DW_Conv2D = tf.keras.layers.Conv2D(self.F, self.K, self.s,
                                             activation='linear', groups=self.F, padding='valid')

        self.built = True
    
    def call(self, inputs):      
        x = self.DW_Conv2D(inputs)      
        x = tf.keras.layers.Reshape((self.N, self.D))(x)
        x = Squash()(x)
        
        return x
    


class FCCaps(tf.keras.layers.Layer):
    """
    Fully-connected caps layer. It exploites the routing mechanism, explained in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing', 
    to create a parent layer of capsules. 
    
    ...
    
    Attributes
    ----------
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    kernel_initilizer: str
        matrix W initialization strategy
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, N, D, kernel_initializer='he_normal', **kwargs):
        super(FCCaps, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]

        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D],initializer=self.kernel_initializer,name='W')
        self.b = self.add_weight(shape=[self.N, input_N,1], initializer=tf.zeros_initializer(), name='b')
        self.built = True
    
    def call(self, inputs, training=None):
        
        u = tf.einsum('...ji,kjiz->...kjz',inputs,self.W)    # u shape=(None,N,H*W*input_N,D)
             
        c = tf.einsum('...ij,...kj->...i', u, u)[...,None]        # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c/tf.sqrt(tf.cast(self.D, tf.float32))
        c = tf.nn.softmax(c, axis=1)                             # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c + self.b
        s = tf.reduce_sum(tf.multiply(u, c),axis=-2)             # s shape=(None,N,D)
        v = Squash()(s)       # v shape=(None,N,D)
        
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D
        }
        base_config = super(FCCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Length(tf.keras.layers.Layer):
    """
    Compute the length of each capsule n of a layer l.
    ...
    
    Methods
    -------
    call(inputs)
        compute the length of each capsule
    """

    def call(self, inputs, **kwargs):
        """
        Compute the length of each capsule
        
        Parameters
        ----------
        inputs: tensor
           tensor with shape [None, num_capsules (N), dim_capsules (D)]
        """
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), - 1) + tf.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config



class Mask(tf.keras.layers.Layer):
    """
    Mask operation described in 'Dynamic routinig between capsules'.
    
    ...
    
    Methods
    -------
    call(inputs, double_mask)
        mask a capsule layer
        set double_mask for multimnist dataset
    """
    def call(self, inputs, double_mask=None, **kwargs):
        if type(inputs) is list:
            if double_mask:
                inputs, mask1, mask2 = inputs
            else:
                inputs, mask = inputs
        else:  
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            if double_mask:
                mask1 = tf.keras.backend.one_hot(tf.argsort(x,direction='DESCENDING',axis=-1)[...,0],num_classes=x.get_shape().as_list()[1])
                mask2 = tf.keras.backend.one_hot(tf.argsort(x,direction='DESCENDING',axis=-1)[...,1],num_classes=x.get_shape().as_list()[1])
            else:
                mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        if double_mask:
            masked1 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask1, -1))
            masked2 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask2, -1))
            return masked1, masked2
        else:
            masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
            return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # generation step
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
    
