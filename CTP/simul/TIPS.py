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
parser.add_argument('--imgFile', type=str, 
                    default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_1e+06.npy')
parser.add_argument('--refFile', type=str, 
                    default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_-1.npy')
parser.add_argument('--paraFile', type=str, 
                    default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/paras_tikh_0.3.npz')
parser.add_argument('--aifFile', type=str, 
                    default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/aif0.npy')
parser.add_argument('--iTest', dest='iTest', type=int, nargs=2, default=[55,70])

# paths
parser.add_argument('--outFile', type=str, default=None)

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--imgNorm', type=float, default=0.15)

parser.add_argument('--windowSize', type=int, nargs=3, default=[11,11,1])
parser.add_argument('--stdTips', type=float, default=0.075)
parser.add_argument('--stdDist', type=float, default=3)


# In[4]:


if sys.argv[0] != 'TIPS.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--device', '0',
                              '--imgFile', '/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_200000.npy',
                              '--stdTips', '0.075',
                              '--stdDist', '3',
                              '--iTest', '55', '58',
                              '--outFile', '/home/dwu/trainData/Noise2Noise/train/ctp/simul/tips/test'])
else:
    args = parser.parse_args(sys.argv[1:])

for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[5]:


# load the image data
imgs = (np.load(args.imgFile)[args.iTest[0]:args.iTest[1], ...] - 1) / args.imgNorm
refs = (np.load(args.refFile)[args.iTest[0]:args.iTest[1], ...] - 1) / args.imgNorm


# In[6]:


# load param maps
with np.load(args.paraFile) as f:
    cbf0 = f['cbf'][args.iTest[0]:args.iTest[1], ...]
    cbv0 = f['cbv'][args.iTest[0]:args.iTest[1], ...]
    mtt0 = f['mtt'][args.iTest[0]:args.iTest[1], ...]
    mask = f['mask'][args.iTest[0]:args.iTest[1], ...]
    cbfFac = f['cbfFac']
aif = np.load(args.aifFile) / 1000 / args.imgNorm

maskVessels = np.where(np.max(imgs, -1) > 0.1 / args.imgNorm, 1, 0)
maskVessels *= mask
for i in range(maskVessels.shape[0]):
    maskVessels[i,...] = scipy.ndimage.morphology.binary_dilation(maskVessels[i,...])
mask *= (1-maskVessels)

imgs *= np.tile(mask[...,np.newaxis], (1,1,1,imgs.shape[-1]))


# In[7]:


HYPR_NLM.SetDevice(args.device)


# In[11]:


if args.stdTips < 0:
    ssd = HYPR_NLM.TIPS(imgs[..., np.newaxis,:], args.windowSize, 0, 0, ssdOnly=1)
    ssd = np.mean(ssd, -1)
    stdTips = np.sqrt(np.sum(mask * ssd) / np.sum(mask))
else:
    stdTips = args.stdTips

recon = HYPR_NLM.TIPS(imgs[..., np.newaxis,:], args.windowSize, stdTips, args.stdDist)
recon = recon[..., 0, :] - (recon[..., 0, [0]] + recon[..., 0, [1]]) / 2


# In[13]:


cbf0, cbv0, mtt0 = paramaps.CalcParaMaps(refs - refs[...,[0]], mask, aif=aif, kappa=1, rho=1)
cbf0 *= cbfFac
mtt0 /= cbfFac


# In[14]:


cbf, cbv, mtt = paramaps.CalcParaMaps(recon, mask, aif=aif, kappa=1, rho=1)
cbf *= cbfFac
mtt /= cbfFac
rmse = np.sqrt(np.sum((cbf - cbf0)**2 * mask, (1,2)) / np.sum(mask, (1,2)))
rmseCbv = np.sqrt(np.sum((cbv - cbv0)**2 * mask, (1,2)) / np.sum(mask, (1,2)))
rmseMtt = np.sqrt(np.sum((mtt - mtt0)**2 * mask, (1,2)) / np.sum(mask, (1,2)))
print (rmse, rmseCbv, rmseMtt)
rmseAll = np.array([rmse, rmseCbv, rmseMtt])


# In[37]:


if not os.path.exists(os.path.dirname(args.outFile)):
    os.makedirs(os.path.dirname(args.outFile))
np.save(args.outFile, recon)
np.save(args.outFile + '_rmse', rmseAll)


# In[ ]:




