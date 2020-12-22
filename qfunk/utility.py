#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
"""

import numpy as np




def trans_x(rho,sys,dim):
    """
    
    Parameters
    ----------
    rho:    matrix containing the quantum state that is to be partially transposed
    sys:    list of spaces with respect to which the partial transpose is taken
    dim:    list of dimensions of the spaces rho is defined on 
    
    Requires
    ----------
    numpy as np
    
    Returns
    ----------
    Partial transpose of rho with respect to the systems given in sys. 
	
	"""
	
    le = len(dim)	
    sh = rho.shape
	#array required for the reshaping of rho so that all indeces are stored 
	#in different axes
    arshape = np.append(dim,dim)
	#axes that need to be permuted
    perm = np.append(sys, np.array(sys)+le)
    perm2 = np.append(np.array(sys)+le, sys)
    transAr = np.arange(2*le)
	#permutation vector for the transposition
    transAr[perm] = transAr[perm2]
	#Now, transpose the axis of the reshaped rho, and shape it back to its 
	#original form
    return rho.reshape(arshape).transpose(transAr).reshape(sh)
	



def trace_x(rho,sys,dim):
    """
    
    Parameters
    ----------
    rho:    matrix containing the quantum state that is to be partially traced over
    sys:    list of spaces with respect to which the partial trace is taken
    dim:    list of dimensions of the spaces rho is defined on 
    
    Requires
    ----------
    numpy as np
    
    Returns
    ----------
    Partial trace of rho with respect to the systems given in sys. 
    
    """
    
    #Dimensions of the traced out and the remaining systems
    D = np.prod(dim)
    Dtrace = np.prod(np.array(dim)[sys])
    Dremain = int(D/Dtrace)
    
    #shape required for simple tracing
    shfinal = [Dremain, Dtrace, Dremain, Dtrace]
    
    #parameters required to decompose rho into its subsystems
    le = len(dim)  
    arshape = np.append(dim,dim)
    
    #permutation to permute all spaces that need to be traced out to the right, 
    #so that they can be traced out in one go.
    perm = np.arange(le)
    perm = np.append(np.delete(perm,sys),np.array(sys))
    perm = np.append(perm,perm+le)
    
    #reshape rho, permute all spaces that need to be traced out to the right, 
    #reshape into the form [Dremain, Dtrace, Dremain, Dtrace] and trace out over
    #the Dtrace parts.
    return np.trace(rho.reshape(arshape).transpose(perm).reshape(shfinal), axis1=1, axis2=3)
    

def sys_permute(rho,perm,dim):
    """
    
    Parameters
    ----------
    rho:    matrix containing the quantum state that is to be partially traced over
    perm:   list defining the permutation of spaces
    dim:    list of dimensions of the spaces rho is defined on 
    
    Requires
    ----------
    numpy as np
    
    Returns
    ----------
    Permutation of rho according to the permutation given in perm
	
	"""
	
    le = len(dim)	
    sh = rho.shape
	#array required for the reshaping of rho so that all indeces are stored 
	#in different axes
    arshape = np.append(dim,dim)
	
	#Array for correct permutation 
    P = np.append(np.array(perm),np.array(perm)+le)
    return rho.reshape(arshape).transpose(P).reshape(sh)



def projl(rho,sys,dim):
    """
    
    Parameters
    ----------
    rho:    nxn Matrix
            
    sys :   list of spaces to be traced out and replaced by a maximally mixed state
    
    dim:    list of the dimensions of the subspaces of rho, with prod(dim) = n


    Requires
    -------
    numpy, sys_permute
    
    Returns
    -------
    tr_{sys}(rho) \otimes \ident_{sys}/d_{sys}.
    That is, the spaces given by sys are traced out and replaced by maximally 
    mixed states
    
    """
    dim = np.array(dim)
    le = len(dim)
    sys = np.array(sys)
    #dimension of the traced out spaces
    dimTr = np.prod(dim[sys])
    
    #Find permutation of spaces 
    remainingSpaces = np.delete(np.arange(le),sys)
    perm1 = np.append(sys,remainingSpaces)
    (perm,dummyArr) = (np.arange(le),np.arange(le))
    perm[perm1]=dummyArr
    return 1./dimTr*sys_permute(np.kron(np.eye(dimTr), trace_x(rho,sys,dim)),perm,dim[perm1])



def Lv(rho,dims):
    """
    
    Parameters
    ----------
    rho:    nxn Matrix
            
    
    dim:    list with four elements containing the dimensions of the subspaces 
            of rho, with prod(dim) = n

    
    Requires
    -------
    numpy, projl
    
    Returns
    -------
    Projection of the matrix rho onto the subspace of valid process matrices. 
    Important: Designed for two-party process matrices. Order of spaces is considered
    to be AIAOBIBO.
    Details on the projection operator can be found in arXiv.1506.03776, Eq. (B20)

    """
    return projl(rho,[1],dims) + projl(rho,[3],dims) - projl(rho,[1,3],dims) - projl(rho,[2,3],dims) + projl(rho,[1,2,3],dims) - projl(rho,[0,1],dims) + projl(rho,[0,1,3],dims)




def tn_product(*args):
    """
    Parameters
    ----------
    varargin: elements which are to be tensored
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    tensor product of the elements varargin
    
    """
    if len(args) == 0:
        print('tn_product requires at least one argument. Returned an empty array.')
        return np.array([])
    else:
        result = args[0]
        for Mat in args[1:]:
            result = np.kron(result,Mat)
        return result

def dagger(M):
    """
    Parameters
    ----------
    M: two dimensional matrix
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    transpose conjugate of input matrix M
    """

    return np.transpose(np.conjugate(M))


def eye_like(M, dtype=np.complex128):
    """
    Parameters
    ----------
    M: two dimensional matrix
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    identity matrix with the same dimension as the output dimension of M
    """

    return np.eye(np.shape(M)[0], dtype=dtype)



def _Ejk(d,j,k):
    """
    Returns the zero dxd complex matrix with a 1 at index j,k 

    Parameters
    ----------
    d : positve integer specifying Hilbert space dimension.
    j : row index
    k : column index

    Requires
    -------
    numpy as np

    Returns
    -------
    ejk :  d x d complex numpy array containing a single 1 at [j,k]


    """
    # construct zero complex matrix
    ejk = np.zeros((d,d), dtype=np.complex128)
    ejk[j,k] = 1.0
    return ejk


def gellman_gen(d):
    """
    Constructs a generalised Gellman matrix orthogonal basis set for d x d hermitian matrices. 

    Parameters
    ----------
    d : positive integer specifying Hilbert space dimension.

    Requires
    -------
    numpy as np

    Returns
    -------
    gellman : d^2 x d x d complex numpy array containing spanning set

    Raises
    ------
    ValueError if dimension is non-int or negative
    """

    # basic input parsing
    assert d>1 and type(d) is int, "Dimension must be positive integer greater than 1"

    # preallocate set arry
    gellman = np.empty((d**2, d, d), dtype=np.complex128)

    # iterate through use cases
    ind = 0
    for k in range(1, d):
        for j in range(0, k):
            # create symmetric component
            set_el_sym = _Ejk(d, j, k) + Ejk(d, k, j)

            # create antisymmetric component
            set_el_asym = -1j*(_Ejk(d, j, k) - Ejk(d, k, j)) 

            # add to set
            gellman[ind,:,:] = set_el_sym
            gellman[ind+1,:,:] = set_el_asym

            # step counter
            ind += 2

    # create diagonal elements 
    for l in range(1, d):

        # initialise zero matrix
        diagonal = np.zeros((d,d), dtype=np.complex128)
        coeff = np.sqrt(2/((l)*(l+1)))
        for i in range(0,l):
            diagonal += (_Ejk(d,i,i))  

        diagonal -= l*_Ejk(d,l,l)    

        # add to collection
        gellman[ind,:,:] = coeff*diagonal
        ind += 1

    # add identity to set
    gellman[-1,:,:] = np.eye(d, dtype=np.complex128)

    return gellman




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
    gellman : d^2 x d x d complex numpy array containing spanning set

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
            mub[k,m,:,:] = np.kron(state, dagger(state))

    return mub



def ent_gen(dim, vec=False):
    """
    Generates a maximally entangled bi-partite system each of dimension dim

    Parameters
    ----------
    dim : positive integer specifying Hilbert space dimension
    vec : boolean specifying whether to return entangled state as state vector or density operator

    Requires
    -------
    numpy as np

    Returns
    -------
    ent : d x d complex numpy array corresponding to maximally entangled state or d x 1 state vector of the same


    """

    # pre allocate entangled state array
    ent = np.zeros((dim**2,1),dtype=np.complex128)

    # iterate over each basis element
    for i in range(dim):
        # generate computaional basis element
        comput_el = np.zeros((dim, 1), dtype=np.complex128)
        # assign value 
        comput_el[i] = 1.0

        # add to state
        ent += np.kron(comput_el, comput_el)

    if vec:
        return ent
    else:
        return np.kron(ent, dagger(ent))/dim