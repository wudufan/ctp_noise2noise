#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from ctypes import *
import os
from scipy.ndimage.filters import gaussian_filter


# In[2]:


if __name__ == '__main__':
    lib = cdll.LoadLibrary('./libHYPR_NLM.so')
else:
    lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libHYPR_NLM.so'))


# In[3]:


def SetDevice(device):
    lib.HYPR_NLM_SetDevice.restype = c_int
    return lib.HYPR_NLM_SetDevice(c_int(device))


# In[4]:


# Guided non-local mean and guided filtering
# The target image was filtered by guided non-local mean first then a guided filtering was applied.  
def HYPR_NLM(img, guide, searchSize, kernelSize, kernelStd, d, eps = 1e-6):
    kernel = np.zeros(kernelSize, np.float32)
    kernel[int(kernelSize[0] / 2), int(kernelSize[1] / 2), int(kernelSize[2] / 2)] = 1
    kernel = gaussian_filter(kernel, kernelStd)
    
    res = np.zeros(img.shape, np.float32)
    img = img.astype(np.float32)
    guide = guide.astype(np.float32)
    kernel = kernel.astype(np.float32)
    
    lib.HYPR_NLM.restype = c_int
    err = lib.HYPR_NLM(res.ctypes.data_as(POINTER(c_float)), 
                       img.ctypes.data_as(POINTER(c_float)), 
                       guide.ctypes.data_as(POINTER(c_float)), 
                       kernel.ctypes.data_as(POINTER(c_float)), 
                       c_int(img.shape[2]), c_int(img.shape[1]), c_int(img.shape[0]), 
                       c_int(searchSize[2]), c_int(searchSize[1]), c_int(searchSize[0]), 
                       c_int(kernelSize[2]), c_int(kernelSize[1]), c_int(kernelSize[0]), 
                       c_float(d * d), c_float(eps))
    
    if not err == 0:
        print (err)
    
    return res

# Guided non-local mean
# Pass img to guide for normal non-local mean
# img - Pass a three dimensional array to this. Suggest applying img = np.copy(img, 'C') before passing, so that 
#         the order of elements will be in C order.
# guide - Three dimensional array. guide = np.copy(guide, 'C') is suggested.
# searchSize - Array / list of length 3.
# kernelSize - Array / list of length 3.
# kernelStd - A Gaussian filter was applied to local patches when calculating the distance, 
#       the kernelStd is the std of that Gaussian filter
# d - The standard deviation of the non-local weighting, this is the most important parameter to be tuned.
# eps - not used in NLM, but in HYPR_NLM for regularization during guided filtering
def NLM(img, guide, searchSize, kernelSize, kernelStd, d, eps = 1e-6):
    kernel = np.zeros(kernelSize, np.float32)
    kernel[int(kernelSize[0] / 2), int(kernelSize[1] / 2), int(kernelSize[2] / 2)] = 1
    kernel = gaussian_filter(kernel, kernelStd)
    
    res = np.zeros(img.shape, np.float32)
    img = img.astype(np.float32)
    guide = guide.astype(np.float32)
    kernel = kernel.astype(np.float32)
    
    lib.NLM.restype = c_int
    err = lib.NLM(res.ctypes.data_as(POINTER(c_float)), 
                  img.ctypes.data_as(POINTER(c_float)), 
                  guide.ctypes.data_as(POINTER(c_float)), 
                  kernel.ctypes.data_as(POINTER(c_float)), 
                  c_int(img.shape[2]), c_int(img.shape[1]), c_int(img.shape[0]), 
                  c_int(searchSize[2]), c_int(searchSize[1]), c_int(searchSize[0]), 
                  c_int(kernelSize[2]), c_int(kernelSize[1]), c_int(kernelSize[0]), 
                  c_float(d * d), c_float(eps))
    
    if not err == 0:
        print (err)
    
    return res


# In[1]:


def TIPS(ctp, windowSize, stdTips, stdDist, eps = 1e-6, ssdOnly = 0):
    res = np.zeros(ctp.shape, np.float32)
    ctp = np.copy(ctp, 'C')
    
    lib.TIPS.restype = c_int
    err = lib.TIPS(res.ctypes.data_as(POINTER(c_float)), 
                   ctp.ctypes.data_as(POINTER(c_float)), 
                   c_int(ctp.shape[0]), c_int(ctp.shape[1]), c_int(ctp.shape[2]), c_int(ctp.shape[3]), c_int(ctp.shape[4]), 
                   c_int(int(windowSize[0]/2)), c_int(int(windowSize[1]/2)), c_int(int(windowSize[2]/2)), 
                   c_float(stdTips * stdTips), c_float(stdDist * stdDist), c_float(eps), c_int(ssdOnly))
    
    if not err == 0:
        print (err)
    
    return res


# In[1]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'HYPR_NLM'])


# In[ ]:




