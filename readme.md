# Temporary Repository for Noise2Noise-based CTP Denoising
This is the relevant code for the paper <br>
Wu, D., Ren, H. and Li, Q., 2020. Self-supervised dynamic ct perfusion image denoising with deep neural networks. IEEE Transactions on Radiation and Plasma Medical Sciences.

Note that the code is not fully cleaned. It may be cleaned in the future. 

## Prerequisite
The environment is configured through miniconda, please refer to the env.yml file in the root. 

The noise insertion for the simulation needs one of my legacy python packages for forward and backward projection:<br>
https://github.com/wudufan/ReconNet<br>
This package is being mitigated to a new version so it is not being maintained. 

## Source data
This work was tested using open-source data. For the simulation, we used: <br>
A. Aichert, M. T. Manhart, B. K. Navalpakkam, R. Grimm et al., “A realistic digital phantom for perfusion C-arm CT based on MRI data,” in IEEE Nuclear Science Symposium Conference Record. Institute of Electrical and Electronics Engineers Inc., 2013, pp. 1–2.

For real data, we used: <br>
 O. Maier, B. H. Menze, J. von der Gablentz, L. Hani ¨ et al., “ISLES 2015 - A public evaluation benchmark for ischemic stroke lesion segmentation from multispectral MRI,” Medical Image Analysis, vol. 35, pp. 250–269, Jan 2017.


## Structures

- CTP folder provides the codes for the CTP denoising. 
- HYPR_NLM provides a cuda-based TIPS denoising, it was a nvidia Nsight project. 
- filelist_uncor.npy records the patients that are not temporally filtered in the ISLES dataset. 

### Preprocess
Provides preprocess codes and parameter map calculation codes. It is called by the other codes. 

### simul
Provides codes for the simulation data. The following notebooks should be run in order to generate the source images:

1. GenerateSimulationDataset.ipynb
2. NoiseInsertion.ipynb

The conventional denoising methods uses:

- Gaussian.py
- TIPS.py
- TV.py

The network based approach uses:

- **Training**: TrainFrameToAvg.py (noise2noise), or TrainSupervised.py (supervised)
- **Testing**: TestNetwork.ipynb

Please refer to simul/Scripts for bash scripts that calls the python scripts. 

### Real
Provides codes for the real data. Preprocess.ipynb should be run firstly to generate the training/testing images. Then the rest of the files are the same with that for the simulation.

