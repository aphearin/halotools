#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from .. import build_compression_matrix

from ...custom_exceptions import HalotoolsError

__all__ = ('test_retrieve_sample', )

def test_retrieve_sample():

