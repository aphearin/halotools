
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def match(n1, n2, x1, x2):
    """
    Given two cumulative abundnace functions, n1 and n2, return x1(x2).
    
    Parameters
    ----------
    n1 : 
        primary abundance function
    
    n2: 
        secondary abundance function
    
    x1 : array_like
    
    x2 : array_like
    
    Returns
    -------
    x1(x2) : function
        callable
    """
    
    #calculate abundances for x1
    n = n1(x1)
    
    #invert the secondary abundance function at each x2
    inverted_n2 = interp1d(n2(x2),x2)
    
    #calculate the value of x2 at the abundances of x1
    x2n = inverted_n2(n)
    
    #get x1 as a function of x2
    x1x2 = interp1d(x1,x2)
    
    #extrapolate beyond x1 range using a linear function
    def fitting_func(x, a, b):
        return a*x+b
    
    #use the first 3 and last three tabulated points
    r_slice = slice(-3,None,None)
    l_slice = slice(3,None,None)
    
    #fit the left and right sides
    right_ext = curve_fit(fitting_func,x1[r_slice],x1x2[r_slice])
    left_ext = curve_fit(fitting_func,x1[l_slice],x1x2[l_slice])
    
    def x1x2_func(x):
        """
        given a value of x1, return matched x2
        
        use the interpolated x1x2 func if x is in the range of tabulated x1,
        otherwise, use the extrapolations
        """
        mask_high = (x>x1)
        mask_low = (x<x1)
        mask_in_range = (x>=x1) & (x<=x1)
        
        #initialize the result
        result = np.zeros(len(x))
        result[mask_in_range] = x1x2(x)
        
        result[mask_high] = right_ext(x)
        result[mask_low] = left_ext(x)
        
        return result
    
    return x1x2_func
    