# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:08:45 2019

@author: thyag
"""

import numpy as np

def cost_function(x,y,theta):
    m = len(y)
    J = 0
    h = sum(np.power( ( x.dot(theta) - np.transpose([y])), 2) )
    J = (1.0/(2*m))* h
    return J
