#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
Available functions: TransX, TraceS
"""

import numpy as np

def TransX(rho,sys,dim):
    """
	Function that computes the partial transpose of rho with respect to the 
	systems given in sys. sys needs to be a list (starting with 0 for the 
	first subsystem, even if there is only one system with respect to which 
	the transpose is taken. dims is the array containing the dimensions of 
	the subsystems.
	
	Requires: numpy
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
	



def TraceX(rho,sys,dim):
    """
    Function that computes the partial trace of rho with respect to the 
    systems given in sys. sys needs to be a list (starting with 0 for the 
    first subsystem), even if there is only one system with respect to which 
    the transpose is taken. dim is the array containing the dimensions of 
    the subsystems.
    
    Requires:  numpy
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
    

def Syspermute(rho,perm,dim):
    """
    Function that permutes the subspaces of rho according to the permutation 
    given by perm. dim contains the dimensions of all subsystems.
	
	Requires: numpy
	"""
	
    le = len(dim)	
    sh = rho.shape
	#array required for the reshaping of rho so that all indeces are stored 
	#in different axes
    arshape = np.append(dim,dim)
	
	#Array for correct permutation 
    P = np.append(np.array(perm),np.array(perm)+le)
    return rho.reshape(arshape).transpose(P).reshape(sh)

def LinkProd(C1,C2):
    """
    
    Parameters
    ----------
    Comb1 : list (M,dim1,Names)1 that contains a comb M with subsystems of the 
            size given by dim1, with corresponding names given by the list 
            Names1
            
    Comb2 : list (N,dim2,Names2) that contains a comb N with subsystems of the 
            size given by dim2, with corresponding names given by the list 
            Names2

    Returns
    -------
    List (M*N,dim, Names) containing of the Link product M*N, the dimensions dim
    of the respective subsystems, and the corresponding names of the subsystems

    """
    
    #get combs themselves and the  dimension and names of the subspaces of the two combs
    (Comb1, dim1, Names1) = (np.array(C1[0]),np.array(C1[1]),np.array(C1[2])) 
    (Comb2, dim2, Names2) = (np.array(C2[0]),np.array(C2[1]),np.array(C2[2])) 
    
    #Find spaces they share
    CommSpaces = np.intersect1d(Names1,Names2)
    
    #If they don't share a space, link product amounts to tensor product
    if len(CommSpaces) == 0:
        return (np.kron(Comb1,Comb2), np.append(dim1,dim2), np.append(Names1,Names2))
    
    #Otherwise, perform a proper link product
    else:
        #Combine information in dictionaries for easy access
        (Dict1,Dict2) = ({},{})
        for n in np.arange(len(Names1)):
            Dict1[Names1[n]] = dim1[n]
        for n in np.arange(len(Names1)):
            Dict2[Names2[n]] = dim2[n]
        Dict = {**Dict1, **Dict2}
        
        #Find spaces they don't share
        #
        #Spaces only in Comb1
        Space1 = Names1[~np.isin(Names1,CommSpaces)]
        #Spaces only in Comb2
        Space2 = Names2[~np.isin(Names2,CommSpaces)]
        #Names of the spaces before contraction. This is the order in which the spaces
        #will be arranged before contraction of the combs
        CorrectOrder= np.append(Names1,Space2)
        
        
    
        #Partially transpose the first comb and tensor it with an identity of the 
        #correct size    
        #dimension of the additional identity matrix
        ExDim1sys = [Dict[Sp2] for Sp2 in Space2]
        ExDim1 = np.prod(np.array(ExDim1sys))
        #Positon of the common spaces
        PosComm1 = np.array([np.where(Names1 == label)  for label in CommSpaces]).flatten()
        #Tensor product and partial transposition over common spaces
        Comb1 = np.kron(TransX(Comb1,PosComm1,dim1),np.eye(ExDim1))
        
        #Tensor the second comb with an identity of the 
        #correct size and rearrange the spaces in the correct order
        
        #Dimension of the additional identity matrix
        ExDim2sys = [Dict[Sp1] for Sp1 in Space1]
        ExDim2 = np.prod(np.array(ExDim2sys))
        #tensoring with identity
        Comb2 = np.kron(Comb2,np.eye(ExDim2))
        
        #Find correct permutation of spaces so that everything is ordered in the same way
        CurrentOrder= np.append(Names2,Space1)
        perm = np.array([np.where(CurrentOrder == label) for label in CorrectOrder]).flatten()
        #Find dimensions of spaces of current order
        dimsCurr = np.array([Dict[Curr] for Curr in CurrentOrder]).flatten()
        #permute spaces of Comb2 according to said permutation
    
        Comb2 = Syspermute(Comb2, perm, dimsCurr)
        
        #Position of the common systems and dimensions of the spaces in right order
        sys = np.array([np.where(CorrectOrder == label) for label in CommSpaces]).flatten()
        dims = np.array([Dict[label] for label in CorrectOrder])
        #New list of space labels after contraction 
        CorrectOrder = CorrectOrder[~np.isin(CorrectOrder,CommSpaces)]
        #dimensions of the remaining spaces after contraction
        dimsNew = np.delete(dims,sys)
        
        return (TraceX(np.dot(Comb1,Comb2), sys, dims), dimsNew, CorrectOrder)



