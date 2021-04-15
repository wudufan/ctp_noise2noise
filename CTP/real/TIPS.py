#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import scipy
import glob


# In[2]:


sys.path.append('../../')
import HYPR_NLM.python.HYPR_NLM as HYPR_NLM
sys.path.append('../Preprocess/')
import CTPPreprocess as preprocess
import CalcParaMaps as paramaps


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'TIPS for CTP')
parser.add_argument('--imgDir', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/real/data/')
parser.add_argument('--nTest', type=int, default=5)

# paths
parser.add_argument('--outFile', type=str, default=None)

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--imgNorm', type=float, default=0.15)

parser.add_argument('--windowSize', type=int, nargs=3, default=[11,11,1])
parser.add_argument('--stdTips', type=float, default=0.075)
parser.add_argument('--stdDist', type=float, default=3)


# In[12]:


if sys.argv[0] != 'TIPS.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--device', '0',
                              '--stdTips', '0.075',
                              '--stdDist', '2',
                              '--nTest', '5',
                              '--outFile', '/home/dwu/trainData/Noise2Noise/train/ctp/real/tips/TIPS_sigma_0.075_2.npy'])
else:
    args = parser.parse_args(sys.argv[1:])

for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[13]:


# load the image data
imgs = np.load(os.path.join(args.imgDir, 'imgs4d.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
masks = np.load(os.path.join(args.imgDir, 'mask.npy'), allow_pickle=True)[-args.nTest:]
aifs = np.load(os.path.join(args.imgDir, 'aif.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
vofs = np.load(os.path.join(args.imgDir, 'vof.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
cbfFacs = np.load(os.path.join(args.imgDir, 'cbfFac.npy'), allow_pickle=True)[-args.nTest:]


# In[14]:


# mask vessels
for i in range(len(imgs)):
    maskVessel = np.where(np.max(imgs[i], -1) > 0.1 / args.imgNorm, 1, 0)[...,np.newaxis]
    maskVessel *= masks[i]
    for k in range(maskVessel.shape[0]):
        maskVessel[k,...,0] = scipy.ndimage.morphology.binary_dilation(maskVessel[k,...,0])
    masks[i] *= (1 - maskVessel)

for i in range(len(imgs)):
    imgs[i] *= masks[i]


# In[15]:


HYPR_NLM.SetDevice(args.device)


# In[16]:


recons = []
for i in range(len(imgs)):
    print (i, end=',', flush=True)
    img = imgs[i]
    if args.stdTips < 0:
        ssd = HYPR_NLM.TIPS(img[..., np.newaxis,:], args.windowSize, 0, 0, ssdOnly=1)
        ssd = np.mean(ssd, -1)
        stdTips = np.sqrt(np.sum(mask * ssd) / np.sum(mask))
    else:
        stdTips = args.stdTips
    
    recon = HYPR_NLM.TIPS(img[..., np.newaxis,:], args.windowSize, stdTips, args.stdDist)
    recon = recon[..., 0, :] - (recon[..., 0, [0]] + recon[..., 0, [1]]) / 2
    recons.append(recon)


# In[18]:


if sys.argv[0] != 'TIPS.py':
    cbf, cbv, mtt = paramaps.CalcParaMaps(recons[2], masks[2], vof = vofs[2], aif=aifs[2])
    cbf *= cbfFacs[2]
    plt.imshow(cbv[7,...] * masks[2][7,...,0], 'jet', vmin=0, vmax=5)


# In[12]:


if not os.path.exists(os.path.dirname(args.outFile)):
    os.makedirs(os.path.dirname(args.outFile))
np.save(args.outFile, recons)


# In[ ]:




