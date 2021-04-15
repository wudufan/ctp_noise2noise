#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy.ndimage


# In[3]:


def BiasCorrection(recon, src, mask):
    reconFiltered = scipy.ndimage.gaussian_filter(recon, (0, 2, 2, 0))
    srcFiltered = scipy.ndimage.gaussian_filter(src, (0, 2, 2, 0))
    bcRecon = np.zeros_like(recon)
    for iSlice in range(reconFiltered.shape[0]):
        ind = np.where(mask[iSlice, ...].flatten() > 0)
        for iFrame in range(reconFiltered.shape[-1]):
            x = reconFiltered[iSlice,...,iFrame].flatten()[ind]
            y = srcFiltered[iSlice,...,iFrame].flatten()[ind]
            p = np.polyfit(x, y, 1)
            bcRecon[iSlice,...,iFrame] = (p[0] * recon[iSlice,...,iFrame] + p[1]) * mask[iSlice,...,0]
    
    return bcRecon


# In[ ]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'BiasCorrection'])

