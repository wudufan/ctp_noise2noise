#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import scipy.ndimage
import sklearn.cluster


# In[2]:


# get brain mask
def BrainMask(ncct, thLow = 0, thHigh = 120):
    bone = np.where(ncct > thHigh, 1, 0)
    for i in range(bone.shape[-1]):
        bone[...,i] = scipy.ndimage.morphology.binary_dilation(bone[..., i])
        bone[...,i] = scipy.ndimage.morphology.binary_closing(bone[...,i], iterations = 2)
    
    mask = np.zeros_like(bone)
    for i in range(bone.shape[-1]):
        mask[...,i] = scipy.ndimage.morphology.binary_fill_holes(bone[..., i])
        
        if np.sum(mask[...,i] - bone[...,i]) == 0:
            # touched boundary
            bone[4, :, i] = bone[4, :, i].max()
            bone[-5, :, i] = bone[-5, :, i].max()
            bone[:, 4, i] = bone[:, 4, i].max()
            bone[:, -5, i] = bone[:, -5, i].max()
            mask[...,i] = scipy.ndimage.morphology.binary_fill_holes(bone[..., i])
    
    return mask - bone


# In[3]:


def GetBloodImage(ncct, ctp, thLow = 0, thHigh = 120):
    mask = BrainMask(ncct, thLow, thHigh)
    ctpm = (ctp - ncct[..., np.newaxis]) * mask[..., np.newaxis]
    
    return ctpm, mask


# In[19]:


def GetSliceToAnalyze(ctpm, mask):
    sliceBrainAreas = np.sum(mask, (0,1))
    sliceToAnalyze = np.argmax(sliceBrainAreas)
    slices = [sliceToAnalyze]
    if sliceToAnalyze > 0:
        slices = [sliceToAnalyze - 1] + slices
    
    if sliceToAnalyze < mask.shape[-1] - 1:
        slices = slices + [sliceToAnalyze + 1]
    
    mask0 = mask[..., slices]
    ctp0 = ctpm[..., slices, :]
    
    return ctp0, mask0


# In[20]:


# find in the posterior half of the brain for VOF
# Ref: Kao YH et al. 2014 
# Automatic measurements of arterial input and venous output functions on cerebral 
# computed tomography perfusion images: a preliminary study
def GetVOF(ctp0, nPoints = 15):
    aucs = np.sum(ctp0, -1)
    aucsVof = np.copy(aucs)
    aucsVof[:, int(aucsVof.shape[1]/2):, :] = 0
    sortedAucs = np.argsort(aucsVof.flatten())[::-1]
    inds = np.unravel_index(sortedAucs[:nPoints], aucs.shape)
    vof = np.mean(np.array([ctp0[ix, iy, iz, :] for ix,iy,iz in zip(inds[0], inds[1], inds[2])]), 0)
    
    return vof, inds


# In[21]:


# use kmeans to select AIF
# Ref: Kim Mouridson et al. 2006
# Automatic Selection of Arterial Input Function Using Cluster Analysis
def GetAIF(ctp0, mask0, vof, pauc = 0.1, preg = 0.75, nCluster = 5, thVofAuc = 0.2):
    # get all the aucs
    inds = np.where(mask0.flatten())[0]
    tacs = np.reshape(ctp0, [-1, ctp0.shape[-1]])[inds, :]
    aucs = np.sum(tacs, -1)

    # get top pauc curves
    inds2 = np.argsort(aucs)[::-1]
    nTop = int(len(aucs) * pauc)
    inds = inds[inds2[:nTop]]
    tacs = tacs[inds2[:nTop], :]

    # standardized tacs
    stacs = tacs / np.sum(tacs, -1)[..., np.newaxis]

    # calculate roughness (square sum of second derivatives) keep only the top preg curves
    d1 = stacs[:, :-1] - stacs[:, 1:]
    d2 = d1[:, :-1] - d1[:, 1:]
    roughs = np.sum(d2*d2, -1)
    inds2 = np.argsort(roughs)
    nTop = int(len(roughs) * preg)
    inds = inds[inds2[:nTop]]
    tacs = tacs[inds2[:nTop]]
    
    # keep aucs that are at least 20% of vof auc
    inds2 = np.where(np.sum(tacs, -1) > thVofAuc * np.sum(vof))[0]
    for i in range(100):
        if len(inds2) <= nCluster * 2:
            thVofAuc *= 0.75
            inds2 = np.where(np.sum(tacs, -1) > thVofAuc * np.sum(vof))[0]
        else:
            break
    
    inds = inds[inds2]
    tacs = tacs[inds2]
    
    # standardized tacs
    stacs = tacs / np.sum(tacs, -1)[..., np.newaxis]
    
    # k-means clustering
    for i in range(3):
        km = sklearn.cluster.KMeans(nCluster, random_state=0).fit(np.copy(stacs, 'C'))
        # calculate mean curve in each cluster
        mstacs = np.array([np.mean(stacs[np.where(km.labels_ == i)[0], :], 0) for i in range(nCluster)])
        # select the one with least first order momentum
        moments = np.sum(mstacs * np.arange(mstacs.shape[-1])[np.newaxis, ...], -1)
        inds2 = np.where(km.labels_ == np.argmin(moments))[0]
        inds = inds[inds2]
        tacs = tacs[inds2]
        stacs = tacs / np.sum(tacs, -1)[..., np.newaxis]

        if tacs.shape[0] <= nCluster * nCluster:
            break
    
    # AIF
    aif = np.mean(tacs, 0)
    indsAif = np.unravel_index(inds, ctp0[...,0].shape)
    
    return aif, indsAif


# In[39]:


def NormalizeAifByDeconv(aif, vof, lam = 0.3):
    aif = scipy.ndimage.gaussian_filter(aif, 1, mode='nearest')
    vof = scipy.ndimage.gaussian_filter(vof, 1, mode='nearest')
    
    aif[aif < 0] = 0
    vof[vof < 0] = 0
    
    # build deconvolution matrix from aif
    mat = np.tril(scipy.linalg.circulant(aif))
    
    u, s, vh = np.linalg.svd(mat, compute_uv=True)
    s[s < lam * s[0]] = 0
    invMat = vh.T @ np.linalg.pinv(np.diag(s)) @ u.T
    h = invMat @ vof
    
    scaling = np.sum(h)
    
    return scaling


# In[40]:


def AutoVofAndAif(ctpm, mask, tSigma = 1, nPoints = 10, pauc = 0.1, preg = 0.75, nCluster = 5, thVofAuc = 0.2):
    ctp0, mask0 = GetSliceToAnalyze(ctpm, mask)
    ctpSmoothed = scipy.ndimage.gaussian_filter(ctp0, (0, 0, 0, tSigma), mode='nearest')
    
    vof, indsVof = GetVOF(ctpSmoothed, nPoints)    
    aif, indsAif = GetAIF(ctpSmoothed, mask0, vof, pauc, preg, nCluster, thVofAuc)
    
    vof = np.mean(np.array([ctp0[ix, iy, iz, :] for ix,iy,iz in zip(indsVof[0], indsVof[1], indsVof[2])]), 0)
    aif = np.mean(np.array([ctp0[ix, iy, iz, :] for ix,iy,iz in zip(indsAif[0], indsAif[1], indsAif[2])]), 0)
    
    return vof, aif, indsVof, indsAif, ctp0


# In[41]:


def PlotVofAndAif(vof, aif, indsVof, indsAif, ctp0):
    plt.figure(figsize=[16,16])
    
    img = np.mean(ctp0, -1)
    for i in range(img.shape[-1]):
        plt.subplot(2,2,i+1);
        plt.imshow(img[...,i], 'gray', vmin=0, vmax=50)
        
        inds = np.where(indsVof[2]==i)[0]
        plt.plot(np.array(indsVof[1])[inds], np.array(indsVof[0])[inds], '.')
        
        inds = np.where(indsAif[2]==i)[0]
        plt.plot(np.array(indsAif[1])[inds], np.array(indsAif[0])[inds], '.')

    plt.subplot(224)
    plt.plot(vof, 'o-')
    plt.plot(aif, 'o-')
    plt.legend(['VOF', 'AIF'])
    


# In[42]:


# test
if __name__ == '__main__':
    get_ipython().run_line_magic('matplotlib', 'inline')
    cases = glob.glob('/home/dwu/data/isles/TRAINING/case_*')

#     for case in cases[38:]:
    for case in cases:
        caseNo = os.path.basename(case).split('_')[-1]
        print (caseNo, end=',')
        
#         if caseNo != '61':
#             continue

        ctp = nib.load(glob.glob(os.path.join(case, '*/*4DPWI*.nii'))[0]).get_fdata()
        ncct = nib.load(glob.glob(os.path.join(case, '*/*O.CT.*.nii'))[0]).get_fdata()

        ctpm, mask = GetBloodImage(ncct, ctp)
        vof, aif, indsVof, indsAif, ctp0 = AutoVofAndAif(ctpm, mask)

        PlotVofAndAif(vof, aif, indsVof, indsAif, ctp0)
        plt.show()
#         plt.savefig(os.path.join('AIFs', caseNo+'.png'))
#         plt.close()
        
        break

    get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'CTPPreprocess'])


# In[ ]:




