# EE367_Project_WNNM_denoising
Stanford EE367 (Computational Imaging) Final Project: Implementation of the Weighted Nuclear Norm Minimization for image denoising.
Andrei Kanavalau

This code is the implementation of S. Gu, L. Zhang, W. Zuo and X. Feng, "Weighted Nuclear Norm Minimization with Application to Image Denoising," 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 2862-2869, doi: 10.1109/CVPR.2014.366.

## Code overview
* wnnm_final.py contains the implementation of the WNNM algorithm
* filters_hw2.py contains implementations of bilateral and non-local means filters
* Results_greyscale_one_img.ipynb can be used to apply the different denoising methods to one image
* Results_greyscale_all.ipynb applies the different denoising methods to all the images in test_images_greyscale, saves the resulting images as well as computed metrics
* Parameter_tuning_c_n.ipynb can be used to investigate the effect of parameters c and n on the denoising performance
* Parameter_tuning_K_and_delta an be used to investigate the effect of parameters K and delta on the denoising performance

Test images are in the folder test_images_greyscale while the output of Results_greyscale_one_img and Results_greyscale_all are stored in the folder results_greyscale
