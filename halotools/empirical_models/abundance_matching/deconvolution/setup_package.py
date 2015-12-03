# Licensed under a 3-clause BSD style license
from __future__ import absolute_import

import os
from distutils.extension import Extension
import numpy as np 

PATH_TO_PKG = os.path.relpath(os.path.dirname(__file__))
SOURCES = ["abunmatch_deconvolution_wrapper.pyx", "src/abunmatch_deconvolution.c"]
THIS_PKG_NAME = '.'.join(__name__.split('.')[:-1])

def get_extensions():

    names = [THIS_PKG_NAME + "." + src.replace('.pyx', '') for src in SOURCES]
    sources = [os.path.join(PATH_TO_PKG, srcfn) for srcfn in SOURCES]
    include_dirs = [np.get_include()]
    libraries = []
    extra_compile_args = []
    
    extensions = []
    extensions.append(Extension(name=names[0],
                        sources=sources,
                        include_dirs=include_dirs,
                        libraries=libraries,
                        extra_compile_args=extra_compile_args))

    return extensions

