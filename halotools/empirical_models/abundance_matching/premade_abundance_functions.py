# -*- coding: utf-8 -*-

"""
premade abudnace function objects
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from ...utils.array_utils import custom_len, array_is_monotonic
from ...custom_exceptions import HalotoolsError
from .abundance_function import *
from .schechter_functions import *
from astropy.modeling.models import custom_model

__all__ = ['LiWhite2009','Baldry2011']

class Baldry2011(AbundanceFunctionFromCallable):
    """
    double schechter stellar mass function
    """
    
    def __init__(self, mstar = np.logspace(7.0,13.0,1000)):
        """
        Paramaters
        ----------
        mstar : array_like, optional 
            abscissa sampling the relevant galaxy/halo property range with an appropriate 
            density. Default is np.logspace(7.0,13.0,1000). 
        """
        
        self.publications = ['arXiv:1111.5707']
        
        
        #define components of triple Schechter function
        s1 = schechter(phi0=3.96*10**(-3), x0=10**10.66, alpha=-0.35)
        s2 = schechter(phi0=0.79*10**(-3), x0=10**10.3702, alpha=-1.47)
        
        #define composite model
        s = s1+s2
        
        #define parameters
        params = {'n' : s,
                  'x' : mstar,
                  'use_log10' : True,
                  'abundance_type' : 'differential',
                  'n_increases_with_x' : False}
        
        #initialize super class
        AbundanceFunctionFromCallable.__init__(self, **params)


class LiWhite2009(AbundanceFunctionFromCallable):
    """
    piecewise triple schechter stellar mass function
    """
    
    def __init__(self, mstar = np.logspace(7.0,13.0,1000)):
        """
        Paramaters
        ----------
        mstar : array_like, optional 
            abscissa sampling the relevant galaxy/halo property range with an appropriate 
            density. Default is np.logspace(7.0,13.0,1000). 
        """
        
        self.publications = ['arXiv:0901.0706']
        
        #define interval used for defining piece-wise components
        @custom_model
        def interval(x,x1=0.0,x2=1.0):
            """
            return 1 if x is in the range and 0 otherwise
            """
            x = np.array(x)
            mask = ((x<=x2) & (x>x1))
            result = np.zeros(len(x))
            result[mask]=1.0
            return result
        
        #define components of triple Schechter function
        a = 10**9.33
        b = 10**10.67
        s1 = schechter(phi0=0.01465, x0=10**9.6124, alpha=-1.1309)*interval(x1=-np.inf,x2=a)
        s2 = schechter(phi0=0.01327, x0=10**10.3702, alpha=-0.9004)*interval(x1=a,x2=b)
        s3 = schechter(phi0=0.00446, x0=10**10.7104, alpha=-1.9918)*interval(x1=b,x2=np.inf)
        
        #define composite model
        s = s1+s2+s3
        
        #define parameters
        params = {'n' : s, 
                  'x' : mstar,
                  'use_log10' : True, 
                  'abundance_type' : 'differential',
                  'n_increases_with_x' : False
                  }
        
        #initialize super class
        AbundanceFunctionFromCallable.__init__(self, **params)

