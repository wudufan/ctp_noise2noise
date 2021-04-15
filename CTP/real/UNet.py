#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import tensorflow as tf
import numpy as np
import argparse


# In[4]:


class UNet:
    def __init__(self, imgshape = [256,256,1], scope='unet2d'):
        self.scope = scope
        self.imgshapeIn = imgshape
        self.imgshapeOut = imgshape
        
        self.depth = 4
        self.nFilters = 32
        self.filterSz = [3,3,3]
        
        self.model = 'unet'
        
        self.bn = 0
        self.beta = 0
        
        self.biasKernelSz = 37
        self.biasKernelStd = 6
    
    def AddArgsToArgParser(self, parser):
        parser.add_argument('--scope', dest='scope', type=str, default='unet2d')
        parser.add_argument('--imgshapeIn', dest='imgshapeIn', type=int, nargs='+', default=[256,256,1])
        parser.add_argument('--imgshapeOut', dest='imgshapeOut', type=int, nargs='+', default=[256,256,1])
        
        parser.add_argument('--nFilters', dest='nFilters', type=int, default=32)
        parser.add_argument('--filterSz', dest='filterSz', type = int, nargs='+', default = [3,3,3])
        parser.add_argument('--depth', dest='depth', type=int, default=4)
        
        parser.add_argument('--model', dest='model', type=str, default='unet')
        
        parser.add_argument('--bn', dest='bn', type=int, default=0)
        parser.add_argument('--beta', dest='beta', type=float, default=0)
        
        parser.add_argument('--biasKernelSz', dest='biasKernelSz', type=int, default=37)
        parser.add_argument('--biasKernelStd', dest='biasKernelStd', type=float, default=6)

        return parser
    
    def FromParser(self, args):
        for k in args.__dict__.keys():
            if k in self.__dict__.keys():
                setattr(self, k, args.__dict__[k])
    
    def Normalization(self, x, name = None):
        if self.bn:
            return tf.layers.batch_normalization(x, scale = False, training = self.training, name = name)
        else:
            return x
    
    def ModelUNet(self, x, reuse=False, scope=None):
        if scope is None:
            scope = self.scope
        
        with tf.variable_scope(scope, reuse = reuse):
            encodes = []
            for i in range(self.depth):
                nFilters = self.nFilters * (2**i)
                if i > 0:
                    x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], 2, padding='same', name='conv%d_down'%i)
                    x = tf.nn.relu(self.Normalization(x, 'bn%d_down'%i))
                x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='conv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'bn%d_0'%i))
#                 x = tf.layers.conv2d(x, nFilters, self.filterSz[:2], padding='same', name='conv%d_1'%i)
#                 x = tf.nn.relu(self.Normalization(x, 'bn%d_1'%i))
                encodes.append(x)
            
            for i in range(self.depth - 2, -1, -1):
                nFilters = self.nFilters * (2**i)
                x = tf.layers.conv2d_transpose(x, nFilters, self.filterSz[:2], 2, padding='same', name='tconv%d_up'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_up'%i))
                x = tf.concat((x, encodes[i]), -1)
                x = tf.layers.conv2d_transpose(x, nFilters, self.filterSz[:2], padding='same', name='tconv%d_0'%i)
                x = tf.nn.relu(self.Normalization(x, 'tbn%d_0'%i))
#                 x = tf.layers.conv2d_transpose(x, nFilters, self.filterSz[:2], padding='same', name='tconv%d_1'%i)
#                 x = tf.nn.relu(self.Normalization(x, 'tbn%d_1'%i))
                
            return tf.layers.conv2d(x, self.imgshapeOut[-1], self.filterSz[:2], padding='same', name='conv_final')
    
    def BuildModel(self):
        with tf.variable_scope(self.scope):
            self.x = tf.placeholder(tf.float32, [None] + self.imgshapeIn, 'x')
            self.ref = tf.placeholder(tf.float32, [None] + self.imgshapeOut, 'ref')
            self.mask = tf.placeholder(tf.float32, [None] + self.imgshapeOut, 'mask')
            self.training = tf.placeholder_with_default(False, None, 'training')
        
        self.recon = self.ModelUNet(self.x, scope=self.scope+'_1')
        
        self.loss2 = tf.reduce_sum(self.mask * (self.recon - self.ref)**2) / tf.reduce_sum(self.mask)
        self.loss = self.loss2
    
    def BuildBiasControl(self):
        with tf.variable_scope(self.scope):
            self.biasKernel = tf.placeholder(tf.float32, [self.biasKernelSz, self.biasKernelSz, 1, 1], 'biasKernel')
            self.aux = tf.placeholder(tf.float32, [None] + self.imgshapeOut, 'aux')
        
        self.reconConv = tf.nn.conv2d(self.recon, self.biasKernel, [1,1,1,1], 'SAME')
        self.auxConv = tf.nn.conv2d(self.aux, self.biasKernel, [1,1,1,1], 'SAME')
        self.auxLoss = tf.reduce_sum(self.mask * (self.reconConv - self.auxConv)**2) / tf.reduce_sum(self.mask)
        
        self.loss = self.loss + self.beta * self.auxLoss


# In[5]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'UNet'])


# In[ ]:




