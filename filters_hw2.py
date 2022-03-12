import numpy as np
from skimage.filters import gaussian

# Adapted from HW2 of EE367: Computational imaging winter quarter 2022

def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)

def bilateral2d(img, radius, sigma, sigmaIntensity):
    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    filtSize = (2*radius + 1, 2*radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerVal = imgPad[y+pad, x+pad] # Careful of padding amount!

            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.
            # Don't forget to normalize by the sum of the weights used.
            
            intensityKernel = np.exp(-(imgPad[y+pad-radius:y+pad+radius+1, x+pad-radius:x+pad+radius+1]-centerVal)**2/(2*sigmaIntensity**2))

            Wp = np.sum(intensityKernel*spatialKernel)
            
            out[y, x] = 1/Wp*np.sum(imgPad[y+pad-radius:y+pad+radius+1, x+pad-radius:x+pad+radius+1]*intensityKernel*spatialKernel)
    return out

def nonlocalmeans(img, searchWindowRadius, averageFilterRadius, sigma, nlmSigma):
    # Initialize output to 0
    out = np.zeros_like(img)
    # Pad image to reduce boundary artifacts
    pad = averageFilterRadius + searchWindowRadius
    imgPad = np.pad(img, pad)
    imgPad = imgPad[..., pad:-pad] # Don't pad third channel

    # Smoothing kernel
    filtSize = (2*averageFilterRadius + 1, 2*averageFilterRadius + 1)
    kernel = fspecial_gaussian_2d(filtSize, sigma)
    # Add third axis for broadcasting
    kernel = kernel[:, :, np.newaxis]
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerPatch = imgPad[y+pad-averageFilterRadius:y+pad+averageFilterRadius+1,
                                 x+pad-averageFilterRadius:x+pad+averageFilterRadius+1,
                                 :]
            # Go over a window around the current pixel, compute weights
            # based on difference of patches, sum the weighted intensity
            # Hint: Do NOT include the patches centered at the current pixel
            # in this loop, it will throw off the weights
            searchWindowImage = imgPad[y+pad-searchWindowRadius:y+pad+searchWindowRadius+1,
                                 x+pad-searchWindowRadius:x+pad+searchWindowRadius+1,
                                 :]
            weights = np.zeros((2*searchWindowRadius+1, 2*searchWindowRadius+1, 1))
            
            for i in range(2*searchWindowRadius+1):
                otherPatch = imgPad[y+pad-searchWindowRadius-averageFilterRadius:y+pad+searchWindowRadius+averageFilterRadius+1,
                                    x+pad-searchWindowRadius+i-averageFilterRadius:x+pad-searchWindowRadius+i+averageFilterRadius+1,
                                    :]
                
                indexer = np.arange((2*averageFilterRadius+1)**2)[None, :] + (2*averageFilterRadius+1)*np.arange(otherPatch.shape[0]-2*averageFilterRadius)[:, None]
                
                otherPatch = otherPatch.flatten()
                
                otherPatch = np.reshape(otherPatch[indexer],(otherPatch[indexer].shape[0],2*averageFilterRadius + 1,2*averageFilterRadius + 1,1))
                
                weights[:,i] = np.exp(-np.sum(kernel*(centerPatch-otherPatch)**2,axis=(1,2))/(2*nlmSigma**2))
            
            weights[searchWindowRadius,searchWindowRadius] = 0
            weights[searchWindowRadius,searchWindowRadius] = np.max(weights)
            weights = weights/np.sum(weights)
            
            # This makes it a bit better: Add current pixel as well with max weight
            # computed from all other neighborhoods.
            max_weight = np.max(weights)

            out[y, x, :] = np.sum(weights*searchWindowImage)
    return out