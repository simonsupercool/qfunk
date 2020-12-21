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
	



def TraceX(rho,sys,dim):
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
    

def Syspermute(rho,perm,dim):
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

def LinkProd(C1,C2):
    """
    
    Parameters
    ----------
    C1 : list (M,dim1,Names)1 that contains a comb M with subsystems of the 
            size given by dim1, with corresponding names given by the list 
            Names1
            
    C2 : list (N,dim2,Names2) that contains a comb N with subsystems of the 
            size given by dim2, with corresponding names given by the list 
            Names2
            
    Requires
    -------
    numpy as np
    TraceX
    TransX

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
    
        
        
def LinkProd_Class(C1,C2):
    """
    ###Same as Link product, but written for when C1 and C2 are classes
    
    Parameters
    ----------
    C1,C2 : Comb class objects
   
    Requires 
    --------
    numpy as np
    TraceX
    TransX

    Returns
    -------
    Comb class object containing the Link product of the two combs as self.mat, 
    the dimensions of the spaces of the resulting comb as self.dims, and the labels
    of the spaces of the resulting comb as self.spaces
    
    
    """
    
    #get combs themselves and the  dimension and names of the subspaces of the two combs
    (Comb1, dim1, Names1) = (np.array(C1.mat),np.array(C1.dims),np.array(C1.spaces)) 
    (Comb2, dim2, Names2) = (np.array(C2.mat),np.array(C2.dims),np.array(C2.spaces)) 
    
    #Find spaces they share
    CommSpaces = np.intersect1d(Names1,Names2)
    
    #If they don't share a space, link product amounts to tensor product
    if len(CommSpaces) == 0:
        return Comb(np.kron(Comb1,Comb2), np.append(dim1,dim2), np.append(Names1,Names2))
    
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
        return Comb(TraceX(np.dot(Comb1,Comb2), sys, dims), dimsNew, CorrectOrder)

    



class Comb():
    """
    Class that contains comb objects and functions that are useful on them
    
    Attributes
    ----------
    self.mat:       overall matrix of the comb
    self.dims:      dimension of the spaces the comb is defined on
    self.spaces:    labels of the spaces the comb is defined on
    
    Functions
    ----------
    relabel:        Allows one to change all the labels of the spaces
    relabelInd:     Allows one to relabel individual spaces
    reduce:         Allows one to trace out particular spaces
    
    Requires 
    --------
    numpy as np
    TraceX
    TransX    
    
    """
    
    def __init__(self,mat,dims,spaces):
        self.mat = mat
        self.dims = dims
        self.spaces = spaces
        
        
    def relabel(self,newlab):
        """
        Parameters
        ----------
        newlab :    list that contains the new labels of the spaces the comb is 
                    defined on            
        
        Returns
        -------
        Comb object with new labels given by the list newlab
        
        Requires
        -------
        numpy as np
        TraceX

        """
        #Check if new labels have the right length
        if len(newlab) != len(self.spaces):
            print('List of new labels does not have the correct length. Nothing has been changed')
        else:
            self.spaces = newlab
    
    def relabelInd(self, oldlab,newlab):
        """
        Parameters
        ----------
        oldlab :    list with old labels that are to be exchanged  
        
        newlab:     list with new labels to replace the ones in oldlab
        
        Returns
        -------
        Comb object with the labels in oldlab replaced by the labels in newlab
        
        Requires
        -------
        numpy as np

        """
        for old in np.arange(len(oldlab)):
            #position of the spaces that are to be exchanged
            ind = np.array(np.where(np.array(self.spaces) == oldlab[old])).flatten()
            if len(ind)!=1:
                print('Something went wrong with the space labels. Nothing has been changed')
            else:
                self.spaces[np.array(np.where(np.array(self.spaces) == oldlab[old])).flatten()[0]] = newlab[old]
                
    def reduce(self,reducspaces):
        """
        Parameters
        ----------
        reducspaces :   list with labels of spaces that should be traced out  
        
        
        Returns
        -------
        Comb object with the spaces in reducspaces traced out
        
        Requires
        -------
        numpy as np
        TraceX

        """
        sys = []
        for n in np.arange(len(reducspaces)):
            sys.append(np.array(np.where(np.array(self.spaces) == reducspaces[n])).flatten())
            print(sys)
        sys = np.array(sys).flatten()
        self.mat = TraceX(self.mat,sys,self.dims)
        print(self.mat)
        self.spaces = np.delete(np.array(self.spaces),sys)
        print(self.spaces)
        self.dims = np.delete(np.array(self.dims),sys)
        print(self.dims)
        

def ProjL(rho,sys,dim):
    """
    
    Parameters
    ----------
    rho:    nxn Matrix
            
    sys :   list of spaces to be traced out and replaced by a maximally mixed state
    
    dim:    list of the dimensions of the subspaces of rho, with prod(dim) = n


    Requires
    -------
    numpy, Syspermute
    
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
    return 1./dimTr*Syspermute(np.kron(np.eye(dimTr), TraceX(rho,sys,dim)),perm,dim[perm1])


def Lv(rho,dims):
    """
    
    Parameters
    ----------
    rho:    nxn Matrix
            
    
    dim:    list with four elements containing the dimensions of the subspaces 
            of rho, with prod(dim) = n

    
    Requires
    -------
    numpy, ProjL
    
    Returns
    -------
    Projection of the matrix rho onto the subspace of valid process matrices. 
    Important: Designed for two-party process matrices. Order of spaces is considered
    to be AIAOBIBO.
    Details on the projection operator can be found in arXiv.1506.03776, Eq. (B20)

    """
    return ProjL(rho,[1],dims) + ProjL(rho,[3],dims) - ProjL(rho,[1,3],dims) - ProjL(rho,[2,3],dims) +ProjL(rho,[1,2,3],dims) - ProjL(rho,[0,1],dims) + ProjL(rho,[0,1,3],dims)




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


def randRho(n):
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

def TnProduct(*args):
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
        print('TnProduct requires at least one argument. Returned an empty array.')
        return np.array([])
    else:
        result = args[0]
        for Mat in args[1:]:
            result = np.kron(result,Mat)
        return result

def Wocb():
    """
    
    Parameters
    ----------
    none
    
    Requires
    -------
    numpy as np
    TnProduct

    Returns
    -------
    Process matrix W_{OCB} defined in arxiv.1105.4464, Eq. (7)
    Hilbert space order is AIAOBIBO
    
    
    """""
    
    #Preliminaries
    ident = np.eye(2)
    sigmaX = np.array([[0, 1], [1,0]])
    sigmaZ = np.array([[1, 0], [0,-1]]);
    
    #Definition of Wocb according to arxiv.1105.4464, Eq. (7)
    return 1./4*(TnProduct(ident,ident,ident,ident) + 1/np.sqrt(2)*(TnProduct(ident,sigmaZ,sigmaZ,ident) + TnProduct(sigmaZ,ident,sigmaX,sigmaZ)))



def ChoitoVec(C,dims):
    """
    Parameters
    ----------
    C:      matrix, corresponding to the Choi state of a map
    dims:   list [dIn,dOut] that contains the input and output dimensions of the map
    
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    Matrix that corresponds to the vectorized version of C
    #Should make sure we mention what type of vectorization we mean
    
	"""
	
    le = len(dims)	
    sh = C.shape
	#array required for the reshaping of rho so that all indeces are stored 
	#in different axes
    arshape = np.append(dims,dims)
	#axes that need to be permuted
    transAr = np.arange(2*le)
    perm2 = [0,2,1,3]
    
	#permutation vector for the transposition
    transAr[[0,1,2,3]] = transAr[perm2]
	#Now, transpose the axis of the reshaped rho, and shape it back to its 
	#original form
    return C.reshape(arshape).transpose(transAr).reshape(sh)
    
    
    
    