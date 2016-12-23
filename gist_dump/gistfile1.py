 def gaussian_density_batch(x, mean, stddev, correlation, compute_derivatives=False): 
  """
  Compute the Gaussian density at x for a 2D normal distribution with parameters mean, stddev, correlation.
  
  This works simultaneously on a batch of inputs.  The inputs should have dimensions: 
    x.shape = (n, 1, 2)
    mean.shape = stddev.shape = (n, m, 2)
    correlation.shape = (n, m, 1)
  where n*m is the number of different Gaussian density functions that we want to evaluate, on n input points x.  
  So the same input x is plugged into the density for m Gaussian pdfs.  (This is convenient for evaluating a 
  mixture over m Gaussians.)  
  The result is an array of probability densities, of shape (n, m).  
  
  If compute_derivatives=True, then it will also compute the derivatives of the logarithm of the density w.r.t. 
  each of the distribution parameters, and return (dmean, dstddev, dcorrelation).  
  However, NOTE that dstddev is actually the derivative with respect to log(stddev), and 
  dcorrelation is actually the derivative with respect to tanh^{-1}(correlation).  
  """
  smooth_eps = 1e-10
  n, m, _ = mean.shape
    
  offset = (x - mean) / (stddev + smooth_eps)
  Z = (offset[:,:,:1] - offset[:,:,1:])**2 + 2 * (1-correlation) * offset[:,:,:1] * offset[:,:,1:]
  # since correlation is always in the range [-1,1] (it comes from a tanh), Z mathematically should be >= 0
  #  however, numerical errors can make it slightly negative (e.g., I saw 5e-7), which could make the exponential 
  #  overflow
  np.maximum(Z, 0, out=Z)
  density = np.exp(-Z / (2 * (1-correlation**2) + smooth_eps)) / (2 * np.pi * stddev[:,:,:1] * stddev[:,:,1:] * np.sqrt(1-correlation**2 + smooth_eps) + smooth_eps)
    
  if not compute_derivatives: 
    return density, None
  
  # the equations for the derivatives are on p.20 of Graves, I have vectorized them here
  C = 1 / (1 - correlation**2 + smooth_eps)
  dmean = C / (stddev + smooth_eps) * (offset - correlation * offset[:,:,[1,0]])
  dstddev = offset * stddev * dmean - 1
  dcorrelation = offset[:,:,:1] * offset[:,:,1:] + correlation * (1 - C * Z)
    
  return density, (dmean, dstddev, dcorrelation)