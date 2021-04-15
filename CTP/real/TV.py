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
parser.add_argument('--imgDir', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/real/data/')
parser.add_argument('--nTest', type=int, default=5)

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
    args = parser.parse_args(['--nTest', '1',
                              '--betas', '2e-3', '2e-3', '1e-4',
                              '--outFile', '/home/dwu/trainData/Noise2Noise/train/ctp/real/tv/test'])
else:
    args = parser.parse_args(sys.argv[1:])

for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[5]:


# load the image data
imgs = np.load(os.path.join(args.imgDir, 'imgs4d.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
masks = np.load(os.path.join(args.imgDir, 'mask.npy'), allow_pickle=True)[-args.nTest:]
aifs = np.load(os.path.join(args.imgDir, 'aif.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
vofs = np.load(os.path.join(args.imgDir, 'vof.npy'), allow_pickle=True)[-args.nTest:] / 1000 / args.imgNorm
cbfFacs = np.load(os.path.join(args.imgDir, 'cbfFac.npy'), allow_pickle=True)[-args.nTest:]


# In[6]:


# mask vessels
for i in range(len(imgs)):
    maskVessel = np.where(np.max(imgs[i], -1) > 0.1 / args.imgNorm, 1, 0)[...,np.newaxis]
    maskVessel *= masks[i]
    for k in range(maskVessel.shape[0]):
        maskVessel[k,...,0] = scipy.ndimage.morphology.binary_dilation(maskVessel[k,...,0])
    masks[i] *= (1 - maskVessel)

for i in range(len(imgs)):
    imgs[i] *= masks[i]


# In[7]:


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


# In[91]:


cbfs = []
cbvs = []
mtts = []
ttps = []
for i in range(len(imgs)):
    print ('Image %d'%i)
    aif = np.copy(aifs[i])
    vof = np.copy(vofs[i])
    img = imgs[i]
    
    scaling = preprocess.NormalizeAifByDeconv(aif, vof)
    aif *= scaling
    aif[aif < 0] = 0
    
    # build system matrix
    A = paramaps.CircConvMatrix(aif, 2*len(aif) + 1)
    u, s, vh = np.linalg.svd(A, compute_uv=True)
    ctp = img - (img[...,[0]] + img[...,[1]]) / 2
    exImgs = np.concatenate((ctp, np.zeros(list(ctp.shape[:-1]) + [len(aif)+1])), -1)

    # TV denoising
    residues = []
    for iSlice in range(exImgs.shape[0]):
        print (iSlice, end=': ')
        x = np.zeros_like(exImgs[[iSlice], ...])
        z = np.copy(x)
        t = 0
        for k in range(args.nIters):
            print (k, end=',', flush=True)
            xNew, zNew, t = TTVOneIteration(x[0,...], z[0,...], t, exImgs[iSlice,...], A, s[0], args)
            x[0,...] = xNew
            z[0,...] = zNew
        residues.append(x)
        print ('')
        
    residues = np.concatenate(residues)

    # calculate parametric maps
    cbf = np.max(residues, -1) * 6000 * cbfFacs[i]
    cbv = np.sum(residues, -1) * 100
    mtt = cbv / (cbf + 1e-6) * 60
    ttp = np.argmax((A @ x.reshape((-1, x.shape[-1])).T).T.reshape(x.shape)[...,:ctp.shape[-1]], -1)
    
    cbfs.append(cbf)
    cbvs.append(cbv)
    mtts.append(mtt)
    ttps.append(ttp)


# In[92]:


# save paramaps
if not os.path.exists(os.path.dirname(args.outFile)):
    os.makedirs(os.path.dirname(args.outFile))
np.savez(args.outFile, cbf=cbfs, cbv=cbvs, mtt=mtts, ttp=ttps)


# In[93]:


if sys.argv[0] != 'TV.py':
    plt.figure(figsize=[16,4])
    plt.subplot(141); plt.imshow(cbfs[0][0,...] * masks[0][0,...,0], 'jet', vmin=0,vmax=500)
    plt.subplot(142); plt.imshow(cbvs[0][0,...] * masks[0][0,...,0], 'jet', vmin=0,vmax=4)
    plt.subplot(143); plt.imshow(mtts[0][0,...] * masks[0][0,...,0], 'jet', vmin=0.25,vmax=0.75)
    plt.subplot(144); plt.imshow(ttps[0][0,...] * masks[0][0,...,0], 'jet', vmin=20,vmax=30)


# In[ ]:




