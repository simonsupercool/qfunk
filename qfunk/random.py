#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
Available functions: trans_x, TraceS
"""

import numpy as np


def random_unitary(n):
    """
    Generates a random unitary matrix sampled over Haar measure

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
    
    """
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
    random_unitary
    
    Returns
    -------
    randomly (according to some unitarily invariant measure) generated density 
    matrix of dimension n x n
    
    """
    
    p = np.diag(np.random.rand(n))
    p = p/np.trace(p)
    U = random_unitary(n)
    p = np.dot(np.dot(U,p),np.conjugate(np.transpose(U)))
    return p


def bistochastic_gen(n):
    """
    
    Parameters
    ----------
    n: size of desired matrix
    
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    A randomly generated bistochastic matrix - unclear what the sampling behaviour is however.
    
    """

    # generate iid matrix
    B = np.random.rand(n,n)
    # normalise columns and rows until bistochastic
    biflag = False
    while not biflag:
        for i in range(n):
            B[:,i] = B[:,i]/np.sum(B[:,i]) 

        for j in range(N):
            B[j,:] = B[j,:]/np.sum(B[j,:]) 

        # check if bistochastic matrix else continue with normalisation procedure
        biflag = True
        for i in range(n):
            if not np.isclose(sum(B[:,i]), 1.0):
                biflag = False
                break
        for j in range(n):
            if not np.isclose(sum(B[j,:]), 1.0):
                biflag = False
                break

    return B