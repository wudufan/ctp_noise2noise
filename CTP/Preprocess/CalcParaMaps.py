#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# %matplotlib inline
import glob
import os
import sys
import scipy.linalg
import scipy.ndimage
import skimage.restoration
import scipy.signal


# In[2]:


import CTPPreprocess


# In[8]:


def CircConvMatrix(aif, N = None):
    if N is None:
        N = len(aif) * 2
    elif N < len(aif):
        N = len(aif)
    
    if N > len(aif):
        aif = np.concatenate((aif, np.zeros([N - len(aif)], np.float32)))
    
    return scipy.linalg.circulant(aif)    


# In[15]:


def StandardConvMatrix(aif):
    m = np.zeros([len(aif), len(aif)], np.float32)
    
    m[:, 0] = aif
    for i in range(1, len(aif)):
        m[i:, i] = aif[:-i]
    
    return m


# In[4]:


def CalcResidue(ctp, invMat):
    if invMat.shape[0] > ctp.shape[-1]:
        ctp = np.concatenate((ctp, np.zeros(list(ctp.shape[:-1]) + [invMat.shape[0] - ctp.shape[-1]])), -1)
    x = np.reshape(ctp, [-1, ctp.shape[-1]])
    r = (invMat @ x.T).T
    return np.reshape(r, ctp.shape)


# In[1]:


def CalcParaMaps(tac, mask, method='tikh', lam=0.3, kappa = 0.73, rho = 1.05, vof = None, aif = None, 
                 useCircConv = True, N = None, directCBV = False):
    if aif is None:
        vof, aif, vofInds, aifInds, _ = CTPPreprocess.AutoVofAndAif(tac, mask)
    
    if vof is not None:
        scaling = CTPPreprocess.NormalizeAifByDeconv(aif, vof)
        aif *= scaling
    
    if N is None:
        N = 2 * len(aif) + 1
    
    # normalization
    aif[aif < 0] = 0
    
    # deconvolution matrix
    if useCircConv:
        convMat = CircConvMatrix(aif, N)
    else:
        convMat = StandardConvMatrix(aif)
    u, s, vh = np.linalg.svd(convMat, compute_uv=True)
    
    if method == 'tikh':
        f = s * s / (s * s + lam*lam *s[0]*s[0])
        sinv = np.linalg.pinv(np.diag(s)) @ np.diag(f)
    else:
        # tsvd
        s[s < lam * s[0]] = 0
        sinv = np.linalg.pinv(np.diag(s))
    
    invMat = vh.T @ sinv @ u.T
    
    r = CalcResidue(tac, invMat) * kappa / rho
    
    if useCircConv:
        cbf = np.max(r, -1) * 6000      # ml/100g/min
    else:
        cbf = r[..., 0] * 6000      # ml/100g/min
    
    if directCBV:
        cbv = np.sum(tac, -1) / np.sum(aif) * 100  # ml/100g
    else:
        cbv = np.sum(r, -1) * 100

    mtt = cbv / (cbf + 1e-6) * 60  # second
    
    return cbf, cbv, mtt


# In[5]:


def SplitTimeCurve(ctp):
    ctp1 = np.copy(ctp)
    for i in range(1, ctp1.shape[-1], 2):
        if i == ctp1.shape[-1] - 1:
            ctp1[..., i] = ctp1[..., i-1]
        else:
            ctp1[..., i] = (ctp1[..., i-1] + ctp1[..., i+1]) / 2
    
    ctp2 = np.copy(ctp)
    ctp2[...,0] = ctp2[...,1]
    ctp2[...,-1] = ctp2[...,-2]
    for i in range(2, ctp2.shape[-1]-2, 2):
        ctp2[...,i] = (ctp2[...,i-1] + ctp2[..., i+1]) / 2
    
    return ctp1, ctp2


# In[79]:


if __name__ == '__main__':
    outDir = '../../train/ctp/data/tikh_0.3/'
    outDirs = []
    outDirs.append(os.path.join(outDir, 'cbf/ref'))
    outDirs.append('../../train/ctp/data/mask')
    outDirs.append(os.path.join(outDir, 'cbf/all'))
    outDirs.append(os.path.join(outDir, 'cbf/1'))
    outDirs.append(os.path.join(outDir, 'cbf/2'))

    for outDir in outDirs:
        if not os.path.exists(outDir):
            os.makedirs(outDir)


# In[91]:


if __name__ == '__main__':
    cases = glob.glob('/home/dwu/data/isles/TRAINING/case_*')
    for case in cases:
        caseNo = os.path.basename(case).split('_')[-1]
        print (caseNo, end=',', flush=True)

        if caseNo != '90':
            continue

        ctp = nib.load(glob.glob(os.path.join(case, '*/*4DPWI*.nii'))[0]).get_fdata().astype(np.float32)
        ncct = nib.load(glob.glob(os.path.join(case, '*/*O.CT.*.nii'))[0]).get_fdata().astype(np.float32)
        ref = nib.load(glob.glob(os.path.join(case, '*/*CBF*.nii'))[0]).get_fdata().astype(np.float32)

        # get aif
        ctpm, mask = CTPPreprocess.GetBloodImage(ncct, ctp)
        vof, aif, vofInds, aifInds, _ = CTPPreprocess.AutoVofAndAif(ctpm, mask)
        # normalization
        scaling = CTPPreprocess.NormalizeAifByDeconv(aif, vof)
        aif *= scaling
        aif[aif < 0] = 0

        # deconvolution matrix
        lam = 0.3
        N = 2 * len(aif)
        circConvMat = CircConvMatrix(aif, N)
        u, s, vh = np.linalg.svd(circConvMat, compute_uv=True)

        # tsvd
    #     s[s < lam * s[0]] = 0
    #     sinv = np.linalg.pinv(np.diag(s))

        # tikhonov
        f = s * s / (s * s + lam*lam *s[0]*s[0])
        sinv = np.linalg.pinv(np.diag(s)) @ np.diag(f)

        invMat = vh.T @ sinv @ u.T

        fctp = scipy.ndimage.gaussian_filter(ctpm, (0, 0, 0, 2), mode='nearest')
        fr = CalcResidue(fctp, invMat)
        r = CalcResidue(ctpm, invMat)

        kappa = 0.73 # for hematocrit correction
        rho = 1.05   # brain density
        cbf = CalcCBPFromResidues(r, r) * kappa / rho
        rcbf = CalcCBPFromResidues(fr, fr) * kappa / rho
    #     rcbf = scipy.ndimage.gaussian_filter(cbf, (2,2,0))

        fac = np.sum(ref * mask) / np.sum(rcbf * mask)
        cbf = cbf * fac
        rcbf = rcbf * fac

    #     break

        # split
#         ctp1, ctp2 = SplitTimeCurve(ctpm)

#         interCtp = np.copy(ctp1)
#         interCtp[::2] = ctp2[::2]
#         interCtp[1::2] = ctp1[1::2]

#         fctp = scipy.ndimage.gaussian_filter(ctpm, (2, 2, 0, 0), mode='nearest')
#         fr = CalcResidue(fctp, invMat)
#         rcbf = CalcCBPFromResidues(fr, fr) * kappa / rho * fac

        break

    #     fctp1 = scipy.ndimage.gaussian_filter(ctp1, (2, 2, 0, 1), mode='nearest')
    #     fr1 = CalcResidue(fctp1, invMat)
    #     r1 = CalcResidue(ctp1, invMat)
    #     cbf1 = CalcCBPFromResidues(r1, fr1) * kappa / rho * fac

    #     fctp2 = scipy.ndimage.gaussian_filter(ctp2, (2, 2, 0, 1), mode='nearest')
    #     fr2 = CalcResidue(fctp2, invMat)
    #     r2 = CalcResidue(ctp2, invMat)
    #     cbf2 = CalcCBPFromResidues(r2, fr2) * kappa / rho * fac

    #     np.save(os.path.join(outDirs[0], caseNo), np.copy(np.transpose(ref, (2,1,0)), 'C'))
    #     np.save(os.path.join(outDirs[1], caseNo), np.copy(np.transpose(mask.astype(np.float32), (2,1,0)), 'C'))
    #     np.save(os.path.join(outDirs[2], caseNo), np.copy(np.transpose(cbf, (2,1,0)), 'C'))
    #     np.save(os.path.join(outDirs[2], caseNo+'_fac'), fac.astype(np.float32))
    #     np.save(os.path.join(outDirs[3], caseNo), np.copy(np.transpose(cbf1, (2,1,0)), 'C'))
    #     np.save(os.path.join(outDirs[4], caseNo), np.copy(np.transpose(cbf2, (2,1,0)), 'C'))


# In[12]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'CalcParaMaps'])


# In[ ]:




