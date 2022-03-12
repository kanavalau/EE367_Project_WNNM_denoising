import numpy as np
from scipy.linalg import svd

def wnnm(img, patchRadius, delta, c, K, sigma_n,N_threshold):
    # This function applies weighted nuclear norm minimization based denoising to the imput image img
    # Specify the search window
    searchWindowRadius = patchRadius*3
    
    # Specify the number of iterations for estimating \hat{X}_j
    N_iter = 3
    
    # Specify the width of padding and pad the noisy image
    pad = searchWindowRadius + patchRadius
    imgPad = np.pad(img, pad_width = pad)
    imgPad = imgPad[..., pad:-pad]
    
    # Initialize variables to be iterated over
    xhat_iter = img
    
    for n in range(K):
        # Pad the image for the iteration
        xhat_iter = np.pad(xhat_iter, pad_width = pad)
        xhat_iter = xhat_iter[..., pad:-pad]
        
        # Regularize the image that is denoised during the iteration
        y_iter = xhat_iter + delta*(imgPad - xhat_iter)
        
        # Initialize the matrix to keep track of how many times each pixel has been updated
        pixel_contribution_matrix = np.ones_like(imgPad)
        
        # Identify similar patches and produce the matrix of similar patches
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                # Select the central patch
                centerPatch = y_iter[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                 i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                 :]
                
                # Initialize the vector of distances between patches 
                dists= np.ones(((2*searchWindowRadius+1)**2))
                # Initialize the matrix of patches
                patches = np.zeros(((2*searchWindowRadius+1)**2,(2*patchRadius)**2))
                # Compute distances between patches
                # This is partially vectorized by using indexing to take out patches in a sliding window fashing 
                # out of a vertical slice through the search window
                for k in range(2*searchWindowRadius+1):
                    # Take a vertical slice in the search window
                    otherPatch = y_iter[j:j+2*pad,
                                    i+k:i+k+2*patchRadius,
                                    :]
                    
                    # Determine indices corresponding to patches in a window sliding down the search window
                    indexer = np.arange((2*patchRadius)**2)[None, :] + (2*patchRadius)*np.arange(otherPatch.shape[0]-2*patchRadius+1)[:, None]
                    
                    # Set columns to be patches
                    otherPatch = otherPatch.flatten()
                    otherPatch = np.reshape(otherPatch[indexer],(otherPatch[indexer].shape[0],(2*patchRadius)**2))
                    
                    # Compute distance and store the corresponding patches
                    dists[k*(2*searchWindowRadius+1):(k+1)*(2*searchWindowRadius+1)] = (np.sum((centerPatch.reshape(((2*patchRadius)**2))-otherPatch)**2,axis=1)/(2*patchRadius)**2).flatten()
                    patches[k*(2*searchWindowRadius+1):(k+1)*(2*searchWindowRadius+1),:] = otherPatch
                    
                # Select to N_threshold nearest patches and creat a patch matrix
                indcs = np.argsort(dists)
                Yj = (patches[indcs[:N_threshold],:]).transpose()
                
                # Center the columns
                Yj_means = np.sum(Yj,axis=0)
                Yj_center = Yj - Yj_means
                
                # First iteration need to estimate singular values of Xj
                U,S,V_T = svd(Yj_center, full_matrices=False)
                sing_val = np.sqrt(np.maximum(S**2-N_threshold*sigma_n**2,0))
                
                # Calculate the weights and sinfular values of \hat{X}_j iteratively
                for m in range(N_iter):
                    w = c*np.sqrt(N_threshold)/(sing_val+10**(-6))
                    sing_val = np.diag(np.maximum(S-w,0))
                
                # Compute \hat{X}_j
                Xj_hat_center = U@np.diag(np.maximum(S-w,0))@V_T
                Xj_hat = Xj_hat_center + Yj_means
                
                # Add the estimate of denoised central patch (first column of \hat{X}_j) to the esmated denoised image clipping it to between 0 and 1
                xhat_iter[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                 i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                 :] = xhat_iter[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                 i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                 :] + np.clip(Xj_hat[:,0].reshape((2*patchRadius,2*patchRadius,1)),0,1)
                
                # Keep track of how many times each pixel has been added to
                pixel_contribution_matrix[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                 i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                 :] = pixel_contribution_matrix[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                 i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                 :] + np.ones_like(pixel_contribution_matrix[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                 i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,:])
        
        # Remove the padding and average out contributions to pixels from different patches
        xhat_iter = xhat_iter[pad:-pad,
                    pad:-pad,
                   :]/pixel_contribution_matrix[pad:-pad,
                    pad:-pad,
                   :]
        
    # Produce the final output
    out = xhat_iter
    return out