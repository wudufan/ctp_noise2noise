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
parser = argparse.ArgumentParser(description = 'ctp supervised netwok')
parser.add_argument('--imgFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_1e+06.npy')
parser.add_argument('--refFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/imgs_-1.npy')
parser.add_argument('--paraFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/paras_tikh_0.3.npz')
parser.add_argument('--aifFile', type=str, default='/home/dwu/trainData/Noise2Noise/train/ctp/simul/data/aif0.npy')
parser.add_argument('--nTrain', type=int, default=50)
parser.add_argument('--nTest', type=int, default=15)

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

# match with real data
parser.add_argument('--cutoffThresh', type=float, default=0, help='set every value in the read-in images below this value to the value, used to match real data from ISLES')

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


if sys.argv[0] != 'TrainSupervised.py':
    from IPython import display
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args(['--device', '0',
                              '--nEpochs', '100',
                              '--beta', '25',
                              '--testInterval', '10',
                              '--outputInterval', '10',
                              '--outDir', '/home/dwu/trainData/Noise2Noise/train/ctp/simul/test_train'])
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


# In[12]:


# build bias control kernel
biasControlKernel = scipy.signal.gaussian(net.biasKernelSz, net.biasKernelStd)[..., np.newaxis]
biasControlKernel = biasControlKernel @ biasControlKernel.T
biasControlKernel = (biasControlKernel / np.sum(biasControlKernel))[..., np.newaxis, np.newaxis]


# In[7]:


# load the image data
imgs = np.load(args.imgFile) - 1
refs = np.load(args.refFile) - 1


# In[8]:


# load param files
with np.load(args.paraFile) as f:
    cbf0 = f['cbf']
    cbv0 = f['cbv']
    mtt0 = f['mtt']
    masks = f['mask'][..., np.newaxis]
    cbfFac = f['cbfFac']
aif0 = np.load(args.aifFile) / 1000

maskVessels = np.where(np.max(imgs, -1) > 0.1, 1, 0)[...,np.newaxis]
maskVessels *= masks
for i in range(maskVessels.shape[0]):
    maskVessels[i,...,0] = scipy.ndimage.morphology.binary_dilation(maskVessels[i,...,0])
masks *= (1-maskVessels)

imgs *= np.tile(masks, (1,1,1,imgs.shape[-1]))
refs *= np.tile(masks, (1,1,1,imgs.shape[-1]))


# In[9]:


def ExtractTrainingBatches(imgs, refs, masks, args, batchSize = None):
    if batchSize is None:
        batchSize = args.batchSize
    
    inputImgs = []
    inputAuxs = []
    inputRefs = []
    inputMasks = []
    
    for i in range(batchSize):       
        # first select slice
        # then sample randomly across framew or sample near the peak
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
        refSlice = refs[[iSlice], ...]
        inputRef = np.copy(refSlice[..., [iFrame]])
        
        # extract background image
        indBk = np.random.permutation(2)
        inputBk = np.copy(imgSlice[..., [indBk[0]]])
        refBk = np.copy(refSlice[..., [indBk[1]]])
        
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


# In[10]:


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


# In[11]:


sess = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list='%s'%args.device, 
                                                                      allow_growth=True)))
sess.run(tf.global_variables_initializer())


# In[13]:


np.random.seed(0)

for epoch in range(args.nEpochs):
    nBatches = args.nTrain * imgs.shape[-1]
    
    for iBatch in range(0, nBatches, args.batchSize):
        inputImg, inputRef, inputAux, inputMask = ExtractTrainingBatches(imgs[:args.nTrain,...], 
                                                                         refs[:args.nTrain,...], 
                                                                         masks[:args.nTrain,...], args)
        
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

        if ((k+1) % (args.outputInterval * 5) == 0 or k == int(nBatches / args.batchSize)) and         sys.argv[0] != 'TrainSupervised.py' and epoch >= 0:              

            # get network reconstruction
            testingImgs = imgs[-args.nTest:, ...]
            testingMasks = masks[-args.nTest:, ...]
            reconTest, iSlices = TestSequence(sess, net, testingImgs, args)

            # testing mask
            maskTest = testingMasks[iSlices, ...]
            maskFrame = np.tile(maskTest, (1,1,1,reconTest.shape[-1]))

            # get gaussian reconstruction
            ctp = testingImgs[iSlices, ...] / args.imgNormOut
            ctp = ctp - ctp[...,[0]]
            ctp *= maskFrame
            x = scipy.ndimage.gaussian_filter(ctp, (0,2,2,0), mode='nearest')

            cbf, _, _ = paramaps.CalcParaMaps(reconTest, maskTest, aif=aif0 / args.imgNormOut, rho=1, kappa=1) * cbfFac
            cbfGaussian, _, _ = paramaps.CalcParaMaps(x, maskTest, aif=aif0 / args.imgNormOut, rho=1, kappa=1) * cbfFac

            # get TAC
            reconTac = np.sum(reconTest * maskFrame, (0,1,2)) / np.sum(maskFrame, (0,1,2))
            imgTac = np.sum(ctp * maskFrame, (0,1,2)) / np.sum(maskFrame, (0,1,2))

            iFrame = np.argmax(imgTac)

            display.clear_output()
            plt.figure(figsize=[18,12])
            plt.subplot(231); plt.imshow(reconTest[0, ..., iFrame] * maskTest[0,...,0], 
                                         cmap='gray', vmin=0, vmax=0.9)
            plt.subplot(232); plt.imshow(x[0,...,iFrame] * maskTest[0,...,0], cmap='gray', vmin=0, vmax=0.9)
            plt.subplot(234); plt.imshow(cbf[0,...] * maskTest[0,...,0], cmap='jet', vmin=0, vmax=50)
            plt.subplot(235); plt.imshow(cbfGaussian[0,...] * maskTest[0,...,0], cmap='jet', vmin=0, vmax=50)
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

        reconTest, _ = TestSequence(sess, net, imgs, args, -1)
        maskFrame = np.tile(masks, (1,1,1,reconTest.shape[-1]))

        np.save(os.path.join(tmpDir, 'iodines'), 
                np.copy(np.transpose((reconTest * maskFrame).astype(np.float32), (0,3,1,2)), 'C'))


# In[ ]:





# In[ ]:




