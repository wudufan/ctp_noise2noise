#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import os
import sys
import scipy
import glob


# In[19]:


sys.path.append('../Preprocess/')
import CTPPreprocess as preprocess
import CalcParaMaps as paramaps


# In[20]:


import argparse
parser = argparse.ArgumentParser(description = 'Gaussian for CTP')
parser.add_argument('--imgDir', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/real/data/')
parser.add_argument('--nTest', type=int, default=5)

# paths
parser.add_argument('--outFile', type=str, default=None)
parser.add_argument('--imgNorm', type=float, default=0.15)

parser.add_argument('--std', type=float, default=1)


# In[21]:


if sys.argv[0] != 'Gaussian.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--std', '0',
                              '--nTest', '5',
                              '--outFile', '/home/dwu/trainData/Noise2Noise/train/ctp/real/gaussian/Gaussian_std_0'])
else:
    args = parser.parse_args(sys.argv[1:])

for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[22]:


# load the image data
imgs = np.load(os.path.join(args.imgDir, 'imgs4d.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
masks = np.load(os.path.join(args.imgDir, 'mask.npy'), allow_pickle=True)[-args.nTest:]
aifs = np.load(os.path.join(args.imgDir, 'aif.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
vofs = np.load(os.path.join(args.imgDir, 'vof.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
cbfFacs = np.load(os.path.join(args.imgDir, 'cbfFac.npy'), allow_pickle=True)[-args.nTest:]


# In[23]:


# mask vessels
for i in range(len(imgs)):
    maskVessel = np.where(np.max(imgs[i], -1) > 0.1 / args.imgNorm, 1, 0)[...,np.newaxis]
    maskVessel *= masks[i]
    for k in range(maskVessel.shape[0]):
        maskVessel[k,...,0] = scipy.ndimage.morphology.binary_dilation(maskVessel[k,...,0])
    masks[i] *= (1 - maskVessel)

for i in range(len(imgs)):
    imgs[i] *= masks[i]


# In[24]:


# gaussian denoising
recons = []
for i in range(len(imgs)):
    print (i, end=',', flush=True)
    if args.std > 0:
        reconMask = scipy.ndimage.filters.gaussian_filter1d(masks[i].astype(np.float32), args.std, 1)
        reconMask = scipy.ndimage.filters.gaussian_filter1d(reconMask, args.std, 2)

        recon = scipy.ndimage.filters.gaussian_filter1d(imgs[i], args.std, 1)
        recon = scipy.ndimage.filters.gaussian_filter1d(recon, args.std, 2)

        reconMask[reconMask <= 1e-6] = 1e-6
        recon = recon / reconMask
        recon = recon - (recon[..., [0]] + recon[..., [1]]) / 2

        recons.append(recon)
    else:
        recons.append(imgs[i] - (imgs[i][..., [0]] + imgs[i][..., [1]]) / 2)


# In[25]:


if sys.argv[0] != 'Gaussian.py':
    cbf, cbv, mtt = paramaps.CalcParaMaps(recons[0], masks[0], vof = vofs[0], aif=aifs[0])
    cbf *= cbfFacs[0]
    plt.imshow(cbf[0,...] * masks[0][0,...,0], 'jet', vmin=0, vmax=500)


# In[26]:


if not os.path.exists(os.path.dirname(args.outFile)):
    os.makedirs(os.path.dirname(args.outFile))
np.save(args.outFile, recons)


# In[ ]:




