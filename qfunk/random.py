#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
Available functions: trans_x, TraceS
"""

import numpy as np


def haar_measure(n):
    """
    
    Parameters
    ----------
    n: size of the random unitary that is to be sampled
    
    
    Requires
    -------
    numpy as np
    
    Code taken directly from arXiv.0609050, p. 11
    
    Returns
    -------
    randomly (according to Haar measure) generated unitary matrix of dimension n x n
    
    """""
    z = (np.random.rand(n,n) + 1j*np.random.rand(n,n))/np.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    q = np.multiply(q,ph,q)
    return q


def rand_rho(n):
    """
    
    Parameters
    ----------
    n: size of the random density matrix that is to be sampled
    
    
    Requires
    -------
    numpy as np
    haar_measure
    
    Returns
    -------
    randomly (according to some unitarily invariant measure) generated density 
    matrix of dimension n x n
    
    """
    
    p = np.diag(np.random.rand(n))
    p = p/np.trace(p)
    U = haar_measure(n)
    p = np.dot(np.dot(U,p),np.conjugate(np.transpose(U)))
    return p