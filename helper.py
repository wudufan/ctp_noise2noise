#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob
import os
import skimage.measure
import imageio


# In[2]:


def CalcSSIM(x, y, window = None):
    if window is not None:
        x = (x - window[0]) / (window[1] - window[0])
        y = (y - window[0]) / (window[1] - window[0])
        x[x < 0] = 0
        x[x > 1] = 1
        y[y < 0] = 0
        y[y > 1] = 1
    
    return skimage.measure.compare_ssim(x, y, data_range = 1.0)


# In[3]:


def SavePng(filename, x, window):
    x = (x - window[0]) / (window[1] - window[0]) * 255
    x[x < 0] = 0
    x[x > 255] = 255
    x = x.astype(np.uint8)
    
    x = np.tile(x[..., np.newaxis], [1,1,4])
    x[...,3] = 255
    
    imageio.imwrite(filename, x, format='png')


# In[4]:


def GetMask(reconNet, resolution):
    r = np.sin(reconNet.da * reconNet.nu / 2) * reconNet.dso / resolution
    nx = reconNet.nx
    ny = reconNet.ny
    mask = np.zeros([nx, ny], np.float32)
    for ix in range(nx):
        for iy in range(ny):
            if (ix - nx/2)**2 + (iy - ny/2)**2 <= r**2:
                mask[ix,iy] = 1
    
    return mask

def GetMasks2D(reconNet, resolutions):
    uniqueRes = np.unique(resolutions)
    uniqueMasks = []
    for res in uniqueRes:
        uniqueMasks.append(GetMask(reconNet, res))
    
    masks = np.zeros([len(resolutions), reconNet.nx, reconNet.ny, 1], np.float32)
    for res, mask in zip(uniqueRes, uniqueMasks):
        for iSlice in np.where(resolutions == res)[0]:
            masks[iSlice, ..., 0] = mask
    
    return masks


# In[6]:


def OrderedSubsetsBitReverse(nPrjs, nSubsets):
    if nPrjs % nSubsets != 0:
        raise ValueError('nPrjs must be divided by nSubsets')
    
    # round to 2^n
    order = range(2**(nSubsets-1).bit_length())    
    
    # bit reverse
    width = len(str(bin(order[-1]))[2:])
    binOrder = ['{:0{width}b}'.format(i, width=width) for i in order]
    revOrder = [int(val[::-1],2) for val in binOrder]
    
    # find the numbers less than nSubsets
    revOrder = [r for r in revOrder if r < nSubsets]
    
    newOrder = np.concatenate([np.arange(val, nPrjs, nSubsets) for val in revOrder])
    
    return newOrder


# In[25]:


def ReadData(names, paths, postfixes, resPath):
    resData = LoadResolutionFile(resPath)
    resolutions = []
    dataset = []
    for i, (path, postfix) in enumerate(zip(paths, postfixes)):
        imgs = []
        for name in names:
            img = np.load(os.path.join(path, name+postfix))
            imgs.append(img)
            if i == 0:
                resolutions += [resData[name]] * img.shape[0]
        dataset.append(np.concatenate(imgs, 0))
    
    return dataset, np.array(resolutions)


# In[8]:


def LoadResolutionFile(resFile):
    with np.load(resFile) as f:
        names = f['names']
        res = f['res']
    
    resDictionary = {}
    for i, name in enumerate(names):
        resDictionary[name] = res[i]
    return resDictionary


# In[1]:


def Augmentation(img, option = 0):
    if option == 1:
        return img[:, ::-1, :, :]
    elif option == 2:
        return img[:, :, ::-1, :]
    elif option == 3:
        return img[:, ::-1, ::-1, :]
    else:
        return img
        


# In[1]:


def AugmentationXYZ(img, option = 0):
    if option == 1:
        return img[:, ::-1, :, :, :]
    elif option == 2:
        return img[:, :, ::-1, :, :]
    elif option == 3:
        return img[:, ::-1, ::-1, :, :]
    elif option == 4:
        return img[:, :, :, ::-1, :]
    elif option == 5:
        return img[:, ::-1, :, ::-1, :]
    elif option == 6:
        return img[:, :, ::-1, ::-1, :]
    elif option == 7:
        return img[:, ::-1, ::-1, ::-1, :]
    else:
        return img


# In[3]:


def ExtractPatch(imgList, patchsize, coords = None):
    if coords is None:
        coords = [np.random.randint(imgList[0].shape[i+1] - patchsize[i]) for i in range(len(patchsize))]
    
    patches = [img[:, 
                   coords[0]:coords[0]+patchsize[0], 
                   coords[1]:coords[1]+patchsize[1],
                   coords[2]:coords[2]+patchsize[2], 
                   :] for img in imgList]
    
    return patches


# In[8]:


if __name__ == '__main__':
    import subprocess
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', 'helper'])


# In[ ]:




