"""
Maxout implementation in Theano
"""

# W,b        - Parameters in neural network layer
# activation - Activation function
# maxoutsize - Number of input neurons to maxout units

# Output activation function
output = activation(T.dot(input,W) + b)

# Maxout                                                                
maxout_out = None                                                       
for i in xrange(maxoutsize):                                            
  t = output[:,i::maxoutsize]                                   
  if maxout_out is None:                                              
    maxout_out = t                                                  
  else:                                                               
    maxout_out = T.maximum(maxout_out, t)  