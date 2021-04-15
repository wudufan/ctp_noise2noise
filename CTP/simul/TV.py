#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import scipy
import glob
import prox_tv


# In[2]:


sys.path.append('../Preprocess/')
import CTPPreprocess as preprocess
import CalcParaMaps as paramaps


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'TV for CTP')
parser.add_argument('--imgFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_1e+06.npy')
parser.add_argument('--refFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_-1.npy')
parser.add_argument('--paraFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/paras_tikh_0.3.npz')
parser.add_argument('--aifFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/aif0.npy')
parser.add_argument('--iTest', type=int, nargs=2, default=[55,70])

# paths
parser.add_argument('--outFile', type=str, default=None)
parser.add_argument('--nIters', type=int, default=30)
parser.add_argument('--betas', type=float, nargs=3, default=[5e-4, 5e-4, 5e-4], help='betas in x,y,t directions')
parser.add_argument('--tikh', type=float, default=0.3)

parser.add_argument('--imgNorm', dest='imgNorm', type=float, default=0.15)


# In[4]:


if sys.argv[0] != 'TV.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--imgFile', '/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_1e+06.npy',
                              '--iTest', '55', '57',
                              '--betas', '5e-4', '5e-4', '1e-4',
                              '--outFile', '/home/dwu/trainData/Noise2Noise/train/ctp/simul/tv/test.npz'])
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
#     cbf0 = f['cbf'][args.iTest[0]:args.iTest[1], ...]
#     cbv0 = f['cbv'][args.iTest[0]:args.iTest[1], ...]
#     mtt0 = f['mtt'][args.iTest[0]:args.iTest[1], ...]
    mask = f['mask'][args.iTest[0]:args.iTest[1], ...]
    cbfFac = f['cbfFac']
aif = np.load(args.aifFile) / 1000 / args.imgNorm

maskVessels = np.where(np.max(imgs, -1) > 0.1 / args.imgNorm, 1, 0)
maskVessels *= mask
for i in range(maskVessels.shape[0]):
    maskVessels[i,...] = scipy.ndimage.morphology.binary_dilation(maskVessels[i,...])
mask *= (1-maskVessels)

imgs *= mask[...,np.newaxis]
refs *= mask[...,np.newaxis]


# In[7]:


cbf0, cbv0, mtt0 = paramaps.CalcParaMaps(refs - refs[...,[0]], mask, kappa=1, rho=1, aif=np.copy(aif))
ttp0 = np.argmax(refs, -1)
cbf0 *= cbfFac
mtt0 /= cbfFac


# In[8]:


# TV denoising according to Ruogu Fang et al, "Robust Low-Dose CT Perfusion Deconvolution via Tensor Total-Variation Regularization", TMI 34(7) 2015
# use Tikhnov regularization to keep the system bias consistent
def TTVOneIteration(x, z, t, imgs, A, s0, args):
    tikh = args.tikh * s0
    
    # SGD
    r = x.reshape([-1, x.shape[-1]]).T
    c = imgs.reshape([-1, imgs.shape[-1]]).T
    
    q1 = A @ r - c
    q2 = tikh * r
    
    q1 = A.T @ q1
    q2 = tikh * q2
    
    aq1 = A @ q1
    aq2 = tikh * q2
    
    s = np.sum((q1 + q2)**2) / (np.sum(aq1**2) + np.sum(aq2**2))
    zNew = r - s * (q1 + q2)
    
    zNew = np.reshape(zNew.T, x.shape)
    
    # TV 
    zNew = prox_tv.tvgen(zNew, args.betas, [1,2,3], [1,1,1], n_threads = 8, max_iters = 10)
    
    # acceleration
    tNew = (1 + np.sqrt(1 + 4 * t * t)) / 2
    xNew = zNew + ((t - 1) / tNew) * (zNew - z)
    
    return xNew, zNew, tNew


# In[9]:


aif[aif < 0] = 0
A = paramaps.CircConvMatrix(aif, 2*len(aif) + 1)
ctp = imgs - (imgs[...,[0]] + imgs[..., [1]]) / 2
exImgs = np.concatenate((ctp, np.zeros(list(ctp.shape[:-1]) + [len(aif)+1])), -1)

u, s, vh = np.linalg.svd(A, compute_uv=True)


# In[10]:


residues = []

for iSlice in range(imgs.shape[0]):
    print (iSlice, end=': ')
    x = np.zeros_like(exImgs[[iSlice], ...])
    z = np.copy(x)
    t = 0
    for i in range(args.nIters):
        print (i, end=',', flush=True)
        xNew, zNew, t = TTVOneIteration(x[0,...], z[0,...], t, exImgs[iSlice,...], A, s[0], args)
        x[0,...] = xNew
        z[0,...] = zNew
    residues.append(x)
    print ('')
residues = np.concatenate(residues)
    
cbf = np.max(residues, -1) * 6000 * cbfFac
cbv = np.sum(residues, -1) * 100
mtt = cbv / (cbf + 1e-6) * 60
ttp = np.argmax((A @ x.reshape((-1, x.shape[-1])).T).T.reshape(x.shape)[...,:ctp.shape[-1]], -1)


# In[11]:


rmse = np.sqrt(np.sum((cbf - cbf0)**2 * mask, (1,2)) / np.sum(mask, (1,2)))
print (rmse)


# In[16]:


# save paramaps
if not os.path.exists(os.path.dirname(args.outFile)):
    os.makedirs(os.path.dirname(args.outFile))
np.savez(args.outFile, cbf=cbf, cbv=cbv, mtt=mtt, ttp=ttp, rmse=rmse)


# In[17]:


if sys.argv[0] != 'TV.py':
    plt.figure(figsize=[16,4])
    plt.subplot(141); plt.imshow(cbf[0,...] * mask[0,...], 'jet', vmin=0,vmax=50)
    plt.subplot(142); plt.imshow(cbv[0,...] * mask[0,...], 'jet', vmin=0,vmax=4)
    plt.subplot(143); plt.imshow(mtt[0,...] * mask[0,...], 'jet', vmin=2.5,vmax=7.5)
    plt.subplot(144); plt.imshow(ttp[0,...] * mask[0,...], 'jet', vmin=20,vmax=30)


# In[18]:


if sys.argv[0] != 'TV.py':
    plt.figure(figsize=[16,4])
    plt.subplot(141); plt.imshow(cbf0[0,...] * mask[0,...], 'jet', vmin=0,vmax=50)
    plt.subplot(142); plt.imshow(cbv0[0,...] * mask[0,...], 'jet', vmin=0,vmax=4)
    plt.subplot(143); plt.imshow(mtt0[0,...] * mask[0,...], 'jet', vmin=2.5,vmax=7.5)
    plt.subplot(144); plt.imshow(ttp0[0,...] * mask[0,...], 'jet', vmin=20,vmax=30)


# In[ ]:




