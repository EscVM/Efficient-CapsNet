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



def squash(s):
    """
    Squash activation function presented in 'Dynamic routinig between capsules'.
    ...
    
    Parameters
    ----------
    s: tensor
        input tensor
    """
    n = tf.norm(s, axis=-1,keepdims=True)
    return tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), s)
    

class PrimaryCaps(tf.keras.layers.Layer):
    """
    Create a primary capsule layer with the methodology described in 'Dynamic routing between capsules'.
    ...
    
    Attributes
    ----------
    C: int
        number of primary capsules
    L: int
        primary capsules dimension (number of properties)
    k: int
        kernel dimension
    s: int
        conv stride
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, C, L, k, s, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.k = k
        self.s = s
        
    def build(self, input_shape):    
        self.kernel = self.add_weight(shape=(self.k, self.k, input_shape[-1], self.C*self.L), initializer='glorot_uniform', name='kernel')
        self.biases = self.add_weight(shape=(self.C,self.L), initializer='zeros', name='biases')
        self.built = True
    
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.kernel, self.s, 'VALID')
        H,W = x.shape[1:3]
        x = tf.keras.layers.Reshape((H, W, self.C, self.L))(x)
        x /= self.C
        x += self.biases
        x = squash(x)      
        return x
    
    def compute_output_shape(self, input_shape):
        H,W = input_shape.shape[1:3]
        return (None, (H - self.k)/self.s + 1, (W - self.k)/self.s + 1, self.C, self.L)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'k': self.k,
            's': self.s
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DigitCaps(tf.keras.layers.Layer):
    """
    Create a digitcaps layer as described in 'Dynamic routing between capsules'. 
    
    ...
    
    Attributes
    ----------
    C: int
        number of primary capsules
    L: int
        primary capsules dimension (number of properties)
    routing: int
        number of routing iterations
    kernel_initializer:
        matrix W kernel initializer
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, C, L, routing=None, kernel_initializer='glorot_uniform', **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.C = C
        self.L = L
        self.routing = routing
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None,H,W,input_C,input_L]"
        H = input_shape[-4]
        W = input_shape[-3]
        input_C = input_shape[-2]
        input_L = input_shape[-1]

        self.W = self.add_weight(shape=[H*W*input_C, input_L, self.L*self.C], initializer=self.kernel_initializer, name='W')
        self.biases = self.add_weight(shape=[self.C,self.L], initializer='zeros', name='biases')
        self.built = True
    
    def call(self, inputs):
        H,W,input_C,input_L = inputs.shape[1:]          # input shape=(None,H,W,input_C,input_L)
        x = tf.reshape(inputs,(-1, H*W*input_C, input_L)) #     x shape=(None,H*W*input_C,input_L)
        
        u = tf.einsum('...ji,jik->...jk', x, self.W)      #     u shape=(None,H*W*input_C,C*L)
        u = tf.reshape(u,(-1, H*W*input_C, self.C, self.L))#     u shape=(None,H*W*input_C,C,L)
        
        if self.routing:
            #Hinton's routing
            b = tf.zeros(tf.shape(u)[:-1])[...,None]                       # b shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
            for r in range(self.routing):
                c = tf.nn.softmax(b,axis=2)                                # c shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
                s = tf.reduce_sum(tf.multiply(u,c),axis=1,keepdims=True)   # s shape=(None,1,C,L)
                s += self.biases       
                v = squash(s)                                              # v shape=(None,1,C,L)
                if r < self.routing-1:
                    b += tf.reduce_sum(tf.multiply(u, v), axis=-1, keepdims=True)
            v = v[:,0,...]      # v shape=(None,C,L)
        else:
            s = tf.reduce_sum(u, axis=1, keepdims=True) 
            s += self.biases
            v = squash(s)
            v = v[:,0,...]
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'routing': self.routing
        }
        base_config = super(DigitCaps, self).get_config()
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
    call(inputs)
        mask a capsule layer

    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  
            inputs, mask = inputs
        else:  
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

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
    
