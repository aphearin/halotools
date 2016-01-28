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

__all__ = ['LiWhite2009','BolshoiMpeak']

class LiWhite2009(AbundanceFunctionFromCallable):
    """
    triple schecter stellar mass function
    """
    
    def __init__(self):
        
        self.publications = ['arXiv:0901.0706']
        
        #abscissa to sample stellar mass function
        mstar = np.logspace(8.0,12.0,1000)
        
        #define interval
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
                  'use_log' : True,
                  'type' : 'differential',
                  'n_increases_with_x' : False}
        
        #initialize super class
        super(LiWhite2009, self).__init__(**params)


class BolshoiMpeak(AbundanceFunctionFromCallable):
    """
    super schecter halo+subhalo mpeak function
    """
    
    def __init__(self):
        
        #abscissa to sample stellar mass function
        mpeak = np.logspace(9,16.0,1000)
        
        #define model
        s = super_schechter(phi0=1.4224*10**(-19), x0=10**14.3144, alpha=-1.9341, beta=0.9238)
        
        #define parameters
        params = {'n' : s,
                  'x' : mpeak,
                  'use_log' : True,
                  'type' : 'differential',
                  'n_increases_with_x' : False}
        
        #initialize super class
        super(BolshoiMpeak, self).__init__(**params)



