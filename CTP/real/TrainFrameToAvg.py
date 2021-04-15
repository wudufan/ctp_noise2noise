#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import sys
import scipy
import scipy.signal
import glob


# In[2]:


import UNet

sys.path.append('../Preprocess')
import CTPPreprocess as preprocess
import CalcParaMaps as paramaps

sys.path.append('../../')
import helper


# In[3]:


import argparse
parser = argparse.ArgumentParser(description = 'ctp noise2noise netwok')
parser.add_argument('--imgDir', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/real/data/')
parser.add_argument('--nTrain', type=int, default=17)
parser.add_argument('--nTest', type=int, default=5)

# paths
parser.add_argument('--outDir', type=str, default=None)

# general network training
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--nEpochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batchSize', type=int, default=20)
parser.add_argument('--outputInterval', type=int, default=50)
parser.add_argument('--testInterval', type=int, default=25)

# noise2noise params fo ctp
parser.add_argument('--peakSampleHalfWidth', type=int, default=2)

# data augmentation
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--imgNormIn', type=float, nargs=2, default=[0.15,0.15])
parser.add_argument('--imgOffsetIn', type=float, nargs=2, default=[-1, -1])

parser.add_argument('--imgNormOut', type=float, default=0.025)
parser.add_argument('--imgOffsetOut', type=float, default=0)


# In[4]:


tf.reset_default_graph()
net = UNet.UNet()
parser = net.AddArgsToArgParser(parser)


# In[5]:


if sys.argv[0] != 'TrainFrameToAvg.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--device', '0',
                              '--nEpochs', '100',
                              '--beta', '50',
                              '--testInterval', '10',
                              '--outputInterval', '10',
                              '--outDir', '/home/dwu/trainData/Noise2Noise/train/ctp/real/test_train'])
else:
    args = parser.parse_args(sys.argv[1:])

for k in args.__dict__:
    print (k, args.__dict__[k], sep=': ', flush=True)


# In[6]:


tf.reset_default_graph()
net = UNet.UNet()
net.FromParser(args)
net.imgshapeIn[-1] = net.imgshapeIn[-1] + 1
net.BuildModel()
net.BuildBiasControl()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(args.lr).minimize(net.loss)

saver1 = tf.train.Saver(max_to_keep=5)
saver2 = tf.train.Saver(max_to_keep=100)
if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)


# In[7]:


# build bias control kernel
biasControlKernel = scipy.signal.gaussian(net.biasKernelSz, net.biasKernelStd)[..., np.newaxis]
biasControlKernel = biasControlKernel @ biasControlKernel.T
biasControlKernel = (biasControlKernel / np.sum(biasControlKernel))[..., np.newaxis, np.newaxis]


# In[8]:


# load the image data
imgs = np.load(os.path.join(args.imgDir, 'imgs4d.npy'), allow_pickle=True) / 1000
masks = np.load(os.path.join(args.imgDir, 'mask.npy'), allow_pickle=True)
aifs = np.load(os.path.join(args.imgDir, 'aif.npy'), allow_pickle=True) / 1000
vofs = np.load(os.path.join(args.imgDir, 'vof.npy'), allow_pickle=True) / 1000
cbfs = np.load(os.path.join(args.imgDir, 'cbf0.npy'), allow_pickle=True)
cbfFacs = np.load(os.path.join(args.imgDir, 'cbfFac.npy'), allow_pickle=True)


# In[9]:


# get vessel masks
for i in range(len(imgs)):
    maskVessel = np.where(np.max(imgs[i], -1) > 0.1, 1, 0)[...,np.newaxis]
    maskVessel *= masks[i]
    for k in range(maskVessel.shape[0]):
        maskVessel[k,...,0] = scipy.ndimage.morphology.binary_dilation(maskVessel[k,...,0])
    masks[i] *= (1 - maskVessel)

for i in range(len(imgs)):
    imgs[i] *= np.tile(masks[i], (1,1,1,imgs[i].shape[-1]))


# In[10]:


# pre-process to get target frames
avgs = []
for i in range(len(imgs)):
    avg = np.copy(imgs[i])
    for k in range(1, avg.shape[-1] - 1):
        avg[...,k] = (imgs[i][...,k-1] + imgs[i][...,k+1]) / 2
    avgs.append(avg)


# In[11]:


# bias correction
for i in range(len(avgs)):
    print (i, end=',')
    x = np.sum(avgs[i] * masks[i], (1,2))
    y = np.sum(imgs[i] * masks[i], (1,2))
    p = y / x
    avgs[i] *= p[:, np.newaxis, np.newaxis, :]


# In[33]:


def ExtractTrainingBatches(_imgs, _avgs, _masks, args, batchSize = None):
    if batchSize is None:
        batchSize = args.batchSize
    
    inputImgs = []
    inputAuxs = []
    inputRefs = []
    inputMasks = []
    
    for i in range(batchSize):
        # first select img
        iImg = np.random.randint(len(_imgs))
        imgs = _imgs[iImg]
        avgs = _avgs[iImg]
        masks = _masks[iImg]
        
        # Then select slice
        # Then sample randomly across framew or sample near the peak
        iSlice = np.random.randint(imgs.shape[0])
        if i < batchSize / 2:
            iFrame = np.random.randint(1, imgs.shape[-1] - 1)
        else:
            # find the peak of tac
            tac = np.mean(imgs[iSlice, ...], (0,1))
            iPeak = np.argmax(tac)
            iFrame = np.random.randint(iPeak - args.peakSampleHalfWidth, iPeak + args.peakSampleHalfWidth + 1)
        
        # extract foreground image
        imgSlice = imgs[[iSlice], ...]
        inputImg = np.copy(imgSlice[..., [iFrame]])
        
        # extract reference image
        avgSlice = avgs[[iSlice], ...]
        inputRef = np.copy(avgSlice[..., [iFrame]])
        
        # extract background image
        indBk = np.random.permutation(2)
        inputBk = np.copy(imgSlice[..., [indBk[0]]])
        refBk = np.copy(imgSlice[..., [indBk[1]]])
        
        # get input and output of network
        inputRef = inputRef - refBk
        inputImg = np.concatenate((inputImg, inputBk), -1)
        
        # get aux for bias correction
        inputAux = inputImg[...,[0]] - inputImg[...,[1]]

        inputImgs.append(inputImg)
        inputRefs.append(inputRef)
        inputAuxs.append(inputAux)
        inputMasks.append(masks[[iSlice], ...])
    
    inputImgs = np.concatenate(inputImgs)
    inputRefs = np.concatenate(inputRefs)
    inputAuxs = np.concatenate(inputAuxs)
    inputMasks = np.concatenate(inputMasks)
    
    return inputImgs, inputRefs, inputAuxs, inputMasks


# In[34]:


def TestSequence(sess, net, imgs, args, iSlices = None):
    if iSlices is None:
        iSlices = [np.random.randint(imgs.shape[0])]
    elif iSlices == -1:
        iSlices = list(range(imgs.shape[0]))
    print (iSlices)
    
    imgNormIn = (args.imgNormIn[0] + args.imgNormIn[1]) / 2
    imgOffsetIn = (args.imgOffsetIn[0] + args.imgOffsetIn[1]) / 2
    
    imgs = imgs[iSlices, ...]
    recons = []
    for i in range(imgs.shape[-1]):
        inputImg1 = np.concatenate((imgs[..., [i]], imgs[..., [0]]), -1)
        inputImg2 = np.concatenate((imgs[..., [i]], imgs[..., [1]]), -1)
        
        recon1 = sess.run(net.recon, {net.x: inputImg1 / imgNormIn + imgOffsetIn})
        recon2 = sess.run(net.recon, {net.x: inputImg2 / imgNormIn + imgOffsetIn})
        
        recon = (recon1 + recon2) / 2 - args.imgOffsetOut
        recons.append(recon)
    
    recons = np.concatenate(recons, -1)

    return recons, iSlices


# In[35]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())


# In[ ]:


np.random.seed(0)

for epoch in range(args.nEpochs):
    nBatches = np.sum([img.shape[0] * img.shape[-1] for img in imgs[:args.nTrain]])
    
    for iBatch in range(0, nBatches, args.batchSize):
        inputImg, inputRef, inputAux, inputMask = ExtractTrainingBatches(imgs[:args.nTrain], 
                                                                         avgs[:args.nTrain], 
                                                                         masks[:args.nTrain], args)
        
        # augmentation
        if args.aug:
            argOption = np.random.randint(4)
            inputImg = helper.Augmentation(inputImg, argOption)
            inputRef = helper.Augmentation(inputRef, argOption)
            inputAux = helper.Augmentation(inputAux, argOption)
            inputMask = helper.Augmentation(inputMask, argOption)
        
        # training
        imgOffsetIn = np.random.uniform(args.imgOffsetIn[0], args.imgOffsetIn[1], [inputImg.shape[0], 1, 1, 1])
        imgNormIn = np.random.uniform(args.imgNormIn[0], args.imgNormIn[1], [inputImg.shape[0], 1, 1, 1])
        imgOffsetOut = args.imgOffsetOut
        imgNormOut = args.imgNormOut
        _, loss2, auxLoss, recon = sess.run(
            [train, net.loss2, net.auxLoss, net.recon], 
            {net.x: inputImg / imgNormIn + imgOffsetIn, 
             net.ref: inputRef / imgNormOut + imgOffsetOut,
             net.aux: inputAux / imgNormOut + imgOffsetOut,
             net.biasKernel: biasControlKernel,
             net.mask: inputMask,
             net.training: True})
        recon -= imgOffsetOut
        
        # display
        k = int(iBatch / args.batchSize)
        if (k+1) % args.outputInterval == 0:
            print ('(%d, %d): loss2 = %g, auxLoss = %g'                   %(epoch, k, loss2, net.beta * auxLoss), flush=True)

        if ((k+1) % (args.outputInterval * 5) == 0 or k == int(nBatches / args.batchSize)) and         sys.argv[0] != 'TrainFrameToAvg.py' and epoch >= 0:              

            # get network reconstruction
            # random select an image
            iImg = np.random.randint(len(imgs) - args.nTest, len(imgs))
            testingImgs = imgs[iImg]
            testingMasks = masks[iImg]
            reconTest, iSlices = TestSequence(sess, net, testingImgs, args)

            # testing mask
            maskTest = testingMasks[iSlices, ...]
            maskFrame = np.tile(maskTest, (1,1,1,reconTest.shape[-1]))

            # get gaussian reconstruction
            ctp = testingImgs[iSlices, ...] / args.imgNormOut
            ctp = ctp - ctp[...,[0]]
            ctp *= maskFrame
            x = scipy.ndimage.gaussian_filter(ctp, (0,2,2,0), mode='nearest')

            cbf, _, _ = paramaps.CalcParaMaps(reconTest, maskTest, vof = vofs[iImg] / args.imgNormOut, aif = aifs[iImg] / args.imgNormOut)
            cbfGaussian, _, _ = paramaps.CalcParaMaps(x, maskTest, vof = vofs[iImg] / args.imgNormOut, aif = aifs[iImg] / args.imgNormOut)

            cbf *= cbfFacs[iImg]
            cbfGaussian *= cbfFacs[iImg]

            # get TAC
            reconTac = np.sum(reconTest * maskFrame, (0,1,2)) / np.sum(maskFrame, (0,1,2))
            imgTac = np.sum(ctp * maskFrame, (0,1,2)) / np.sum(maskFrame, (0,1,2))

            iFrame = np.argmax(imgTac)

            display.clear_output()
            plt.figure(figsize=[18,12])
            plt.subplot(231); plt.imshow(reconTest[0, ..., iFrame] * maskTest[0,...,0], 
                                         cmap='gray', vmin=0, vmax=0.9)
            plt.subplot(232); plt.imshow(x[0,...,iFrame] * maskTest[0,...,0], cmap='gray', vmin=0, vmax=0.9)
            plt.subplot(234); plt.imshow(cbf[0,...] * maskTest[0,...,0], cmap='jet', vmin=0, vmax=500)
            plt.subplot(235); plt.imshow(cbfGaussian[0,...] * maskTest[0,...,0], cmap='jet', vmin=0, vmax=500)
            plt.subplot(233); plt.plot(reconTac); plt.plot(imgTac);
            plt.show()
            
    if (epoch + 1) % args.testInterval != 0 and epoch != args.nEpochs - 1:
        saver1.save(sess, os.path.join(args.outDir, '%d'%epoch))
    else:
        saver2.save(sess, os.path.join(args.outDir, '%d'%epoch))
        
        # save intermediate results
        
        print ('Generating intermediate results')
        
        tmpDir = os.path.join(args.outDir, 'tmp')
        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)

        reconTests = []
        for i in range(len(imgs)):
            reconTest, _ = TestSequence(sess, net, imgs[i], args, -1)
            reconTests.append(reconTest * masks[i])

        
        np.save(os.path.join(tmpDir, 'iodines'), reconTests)


# In[ ]:




