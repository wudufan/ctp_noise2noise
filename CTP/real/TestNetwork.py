#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import sys
import scipy
import scipy.signal
import glob


# In[2]:


import UNet

sys.path.append('../Preprocess')
import CTPPreprocess as preprocess
import CalcParaMaps as paramaps


# In[15]:


import argparse
parser = argparse.ArgumentParser(description = 'ctp noise2noise netwok')
parser.add_argument('--imgDir', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/real/data/')
parser.add_argument('--nTest', type=int, default=5)

# paths
parser.add_argument('--checkPoint', type=str, default=None)
parser.add_argument('--outFile', type=str, default=None)

# general network training
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--imgNormIn', type=float, default=0.15)
parser.add_argument('--imgOffsetIn', type=float, default=-1)

parser.add_argument('--imgNormOut', type=float, default=0.025)
parser.add_argument('--imgOffsetOut', type=float, default=0)


# In[16]:


tf.reset_default_graph()
net = UNet.UNet()
parser = net.AddArgsToArgParser(parser)


# In[17]:


if sys.argv[0] != 'TestNetwork.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--device', '0',
                              '--nTest', '1',
                              '--checkPoint', '/home/dwu/trainData/Noise2Noise/train/ctp/simul/supervised_beta_25_N0_200000/99',
                              '--outFile', '/home/dwu/trainData/Noise2Noise/train/ctp/real/supervised/test'])
else:
    args = parser.parse_args(sys.argv[1:])

for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[18]:


tf.reset_default_graph()
net = UNet.UNet()
net.FromParser(args)
net.imgshapeIn[-1] = net.imgshapeIn[-1] + 1
net.BuildModel()

loader = tf.train.Saver()
if not os.path.exists(os.path.dirname(args.outFile)):
    os.makedirs(os.path.dirname(args.outFile))


# In[19]:


# load the image data
imgs = np.load(os.path.join(args.imgDir, 'imgs4d.npy'), allow_pickle=True) / 1000
masks = np.load(os.path.join(args.imgDir, 'mask.npy'), allow_pickle=True)
aifs = np.load(os.path.join(args.imgDir, 'aif.npy'), allow_pickle=True) / 1000
vofs = np.load(os.path.join(args.imgDir, 'vof.npy'), allow_pickle=True) / 1000
cbfs = np.load(os.path.join(args.imgDir, 'cbf0.npy'), allow_pickle=True)
cbfFacs = np.load(os.path.join(args.imgDir, 'cbfFac.npy'), allow_pickle=True)


# In[20]:


# load param files
for i in range(len(imgs)):
    maskVessel = np.where(np.max(imgs[i], -1) > 0.1, 1, 0)[...,np.newaxis]
    maskVessel *= masks[i]
    for k in range(maskVessel.shape[0]):
        maskVessel[k,...,0] = scipy.ndimage.morphology.binary_dilation(maskVessel[k,...,0])
    masks[i] *= (1 - maskVessel)

for i in range(len(imgs)):
    imgs[i] *= masks[i]


# In[21]:


def TestSequence(sess, net, imgs, args, iSlices = None):
    if iSlices is None:
        iSlices = [np.random.randint(imgs.shape[0])]
    elif iSlices == -1:
        iSlices = list(range(imgs.shape[0]))
    print (iSlices)
    
    imgNormIn = args.imgNormIn
    imgOffsetIn = args.imgOffsetIn
    
    imgs = imgs[iSlices, ...]
    recons = []
    for i in range(imgs.shape[-1]):
        print (i, end=',')
        inputImg1 = np.concatenate((imgs[..., [i]], imgs[..., [0]]), -1)
        inputImg2 = np.concatenate((imgs[..., [i]], imgs[..., [1]]), -1)
        
        recon1 = sess.run(net.recon, {net.x: inputImg1 / imgNormIn + imgOffsetIn})
        recon2 = sess.run(net.recon, {net.x: inputImg2 / imgNormIn + imgOffsetIn})
        
        recon = (recon1 + recon2) / 2 - args.imgOffsetOut
        recons.append(recon)
    
    recons = np.concatenate(recons, -1)

    return recons, iSlices


# In[22]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())
loader.restore(sess, args.checkPoint)


# In[24]:


# save intermediate results
print ('Generating results')
if args.nTest > 0:
    testList = np.arange(len(imgs)-args.nTest, len(imgs))
else:
    testList = np.arange(len(imgs))

reconTests = []
maskFrames = []
for i in testList:
    reconTest, _ = TestSequence(sess, net, imgs[i], args, -1)
    reconTests.append(reconTest * masks[i])

np.save(os.path.join(args.outFile), reconTests)


# In[ ]:




