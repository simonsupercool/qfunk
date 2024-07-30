#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
Available functions: trans_x, TraceS
"""

import numpy as np
import qfunk.utility
import scipy.linalg as SLA


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
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)

    q,r = np.linalg.qr(z)
    
    d = np.diagonal(r)
    
    ph = d/np.absolute(d)
    
    q = np.multiply(q,ph,q)
    
    return q


def rand_rho(n, pure=False):
    """
    
    Parameters
    ----------
    n: size of the random density matrix that is to be sampled
    pure: whether to return a random pure state 
    
    Requires
    -------
    numpy as np
    random_unitary
    
    Returns
    -------
    randomly (according to some unitarily invariant measure) generated density 
    matrix of dimension n x n
    
    """

    # check if pure state requested
    if pure:
        # generate equal superposition state
        p = np.ones((n,n))/n
    else:
        # random probability vector as diagonal
        p = np.diag(np.random.rand(n))
        p = p/np.trace(p)
    

    U = random_unitary(n)
    p = np.dot(np.dot(U,p),np.conjugate(np.transpose(U)))
    return p


def bistochastic_gen(n):
    """
    Generates a random n x n bistochastic matrix

    Parameters
    ----------
    n: size of desired matrix
    
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    A randomly generated bistochastic matrix - unclear what the sampling behaviour is however,  
    """

    # generate iid matrix
    B = np.random.rand(n,n)
    # normalise columns and rows until bistochastic
    biflag = False
    while not biflag:
        for i in range(n):
            B[:,i] = B[:,i]/np.sum(B[:,i]) 

        for j in range(n):
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



def MUB_gen(d):
    """
    Generates a maximal MUB in d-dimensional Hilbert space for prime d
    
    Parameters
    ----------
    d : positive integer specifying Hilbert space dimension. Must be prime number

    Requires
    -------
    numpy as np

    Returns
    -------
    mub : d^2 x d x d complex numpy array containing spanning set

    """

    # base constant
    w = np.exp(2*np.pi*1j/d)
    # MUB container
    mub = np.zeros((d+1,d,d,d),dtype=np.complex128)
    # assign computational basis
    for i in range(d):
        mub[0,i,i,i] = 1.0

    # iteratovely construct the MUB
    for k in range(1,d+1):
        for m in range(d):
            state = np.zeros((d,1), dtype=np.complex128)
            for l in range(d):
                el = mub[0,l,:,l].reshape(d,1)
                state += w**(k*(l**2)+m*l) * el/np.sqrt(d)   
            mub[k,m,:,:] = np.kron(state, qfunk.utility.dagger(state))

    return mub



def ent_gen(n, vec=False):
    """
    Generates a maximally entangled bi-partite system each of dimension dim

    Parameters
    ----------
    n : positive integer specifying Hilbert space dimension
    vec : boolean specifying whether to return entangled state as state vector or density operator

    Requires
    -------
    numpy as np

    Returns
    -------
    ent : n x n complex numpy array corresponding to maximally entangled state or n x 1 state vector of the same


    """

    # pre allocate entangled state array
    ent = np.zeros((n**2,1),dtype=np.complex128)

    # iterate over each basis element
    for i in range(n):
        # generate computaional basis element
        comput_el = np.zeros((n, 1), dtype=np.complex128)
        # assign value 
        comput_el[i] = 1.0

        # add to state
        ent += np.kron(comput_el, comput_el)

    if vec:
        return ent
    else:
        return np.kron(ent, qfunk.utility.dagger(ent))/n
    
def random_inst(num_of_el,dim_in,dim_out=None):
    """
    Generates a random instrument in the representation given by Rep

    Parameters
    ----------
    num_of_el:  integer specifying the number of elements of the instrument
    in_dim:     integer specifying the dimension of the input of the map
    out_dim:    integer specifying the dimension of the output of the map
    Rep:        string specifying the representation in which the map is returned

    Requires
    -------
    numpy as np
    scipy.linalg as SLA
    qfunk.utility as qut
    rand_rho

    Returns
    -------
    Inst:   List of num_of_el CP maps that add up to CPTP map

    Desiderata: Enable different representations (currently: Choi Rep)
    """
    #Check if input and output space need to differ
    if dim_out==None:
        dim_out = dim_in
    
    #size of Choi matrices
    dim_tot = dim_in*dim_out
    
    #generate num_of_el random CP maps with trace in (0,1)
    Inst = []
    for n in np.arange(num_of_el):
        tra = np.random.rand()
        Inst.append(tra*rand_rho(dim_tot))
    
    #compute sum of CP, reduced state thereof and inverse
    inst_sum = np.sum([cp for cp in Inst], 0)
    inst_red = qfunk.utility.trace_x(inst_sum, [1],[dim_in,dim_out])
    renormalize = np.kron(SLA.sqrtm(np.linalg.inv(inst_red)),np.eye(dim_out))
    Inst = [np.dot(np.dot(renormalize,cp),renormalize) for cp in Inst]
    return Inst

if __name__ == '__main__':
    Inst = random_inst(2,2)
    print(qfunk.utility.trace_x(Inst[0],[1],[2,2])+qfunk.utility.trace_x(Inst[1],[1],[2,2]))
    print(Inst[0])
    print(Inst[1])
