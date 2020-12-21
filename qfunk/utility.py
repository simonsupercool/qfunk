#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
Available functions: trans_x, TraceS
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