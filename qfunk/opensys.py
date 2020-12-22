#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
Available functions: trans_x, TraceS
"""

import numpy as np

from qfunk.utility import *


def link_prod(C1,C2):
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
    trace_x
    trans_x

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
        Comb1 = np.kron(trans_x(Comb1,PosComm1,dim1),np.eye(ExDim1))
        
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
    
        Comb2 = sys_permute(Comb2, perm, dimsCurr)
        
        #Position of the common systems and dimensions of the spaces in right order
        sys = np.array([np.where(CorrectOrder == label) for label in CommSpaces]).flatten()
        dims = np.array([Dict[label] for label in CorrectOrder])
        #New list of space labels after contraction 
        CorrectOrder = CorrectOrder[~np.isin(CorrectOrder,CommSpaces)]
        #dimensions of the remaining spaces after contraction
        dimsNew = np.delete(dims,sys)
        
        return (trace_x(np.dot(Comb1,Comb2), sys, dims), dimsNew, CorrectOrder)

    


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
    relabel:            Allows one to change all the labels of the spaces
    relabelInd:         Allows one to relabel individual spaces
    reduce:             Allows one to trace out particular spaces
    link:               Allows one to compute the link product with another comb
    dim_of_space:       Returns the dimensions of given spaces of the comb
    dims_of_all_spaces: Returns dimensions of all spaces as dictionary
    
    Requires 
    --------
    numpy as np
    trace_x
    trans_x    
    
    """
    
    def __init__(self,mat,dims,spaces):
        self.mat = mat
        self.dims = dims
        self.spaces = spaces
        #Throw warning if labels appear more than once
        unique, count = np.unique(self.spaces,return_counts=True)
        if sum(count) > len(unique):
            print('Warning: Some of your labels appear more than once. A comb is returned but you WILL definitely run into trouble!')
        
        #Throw warning if spaces don't amount to the correct dimension
        if len((self.mat).flatten()) != (np.prod(np.array(dims)))**2:
            print('Warning: The dimension of the comb does not fit with the given list of dimensions. A comb is returned but you WILL definitely run into trouble!')
    
    
        
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
        trace_x

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
        trace_x

        """
        sys = []
        for n in np.arange(len(reducspaces)):
            sys.append(np.array(np.where(np.array(self.spaces) == reducspaces[n])).flatten())
            print(sys)
        sys = np.array(sys).flatten()
        self.mat = trace_x(self.mat,sys,self.dims)
        print(self.mat)
        self.spaces = np.delete(np.array(self.spaces),sys)
        print(self.spaces)
        self.dims = np.delete(np.array(self.dims),sys)
        print(self.dims)
        
    def link(self,C2,overwrite=False):
        """
        
        Parameters
        ----------
        self
        C2 :    Comb class object
        change: Boolean variable
   
        Requires 
        --------
        numpy as np
        trace_x
        trans_x

        Returns
        -------
        Comb class object containing the Link product of the comb itself and the comb
        C2. Matrix of the resulting comb will be self.mat, the dimensions of the spaces of the         
        resulting comb will be self.dims, and the labels of the spaces of the resulting 
        comb will be self.spaces. If overwrite==True, the original comb will be overwritten.
    
        ###Desiderata:
            - Store combs in different representations
            - define multilink product
            - ....
        """
    
        #get combs themselves and the  dimension and names of the subspaces of the two combs
        (Comb1, dim1, Names1) = (np.array(self.mat),np.array(self.dims),np.array(self.spaces)) 
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
            Comb1 = np.kron(trans_x(Comb1,PosComm1,dim1),np.eye(ExDim1))
            
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
        
            Comb2 = sys_permute(Comb2, perm, dimsCurr)
            
            #Position of the common systems and dimensions of the spaces in right order
            sys = np.array([np.where(CorrectOrder == label) for label in CommSpaces]).flatten()
            dims = np.array([Dict[label] for label in CorrectOrder])
            #New list of space labels after contraction 
            CorrectOrder = CorrectOrder[~np.isin(CorrectOrder,CommSpaces)]
            #dimensions of the remaining spaces after contraction
            dimsNew = np.delete(dims,sys)
            
            #Resulting link product matrix
            LinkMat= trace_x(np.dot(Comb1,Comb2), sys, dims)
            
            #Change matrix in place if demanded
            if overwrite:
                self.mat = LinkMat
                self.dims = dimsNew
                self.spaces = CorrectOrder
            return Comb(LinkMat, dimsNew, CorrectOrder)
        
    def dim_of_space(self,labels):
        """
        
        Parameters
        ----------
        self
        labels: list of labels
           
        Requires 
        --------
        nnumpy as np

        Returns
        -------
        dimension of spaces labeled by the labels in the list labels
    
        
        """
        #get label information from comb
        Names = np.array(self.spaces)
        
        #Make local copy of labels to work with
        lab_cop = labels.copy()
        
        
        #Make sure that none of the labels appears twice, if so, simplify list
        unique, count = np.unique(labels,return_counts=True)
        if sum(count) > len(unique):
            lab_cop = unique
            print('Warning: The provided list of labels contains duplicates. Computation continues with the simplified list', unique)
        
        #Make sure the labels actually all occur, if not, remove those that don't
        contains_check = np.in1d(lab_cop, Names)
        if sum(contains_check) != len(lab_cop):
            if sum(contains_check) == 0:
                print('Warning: None the provided labels actually occured, computation did not return a result')
                return
            else: 
                lab_cop = lab_cop[np.where(contains_check)]
                print('Warning: Not all of the provided labels actually occur Computation contiures with the simplified list', lab_cop )
            
        
        #get dimension information from comb
        dims = np.array(self.dims)
        
        #Make dictionary of spaces and dimensions for easy access
        Dict = {}
        for n in np.arange(len(Names)):
            Dict[Names[n]] = dims[n]
        
        #Print dimensions of the given labels and store in array
        dims_of_labels = []
        for lab in lab_cop:
            dim = Dict[lab]
            print(str(lab) + ':', Dict[lab])
            dims_of_labels.append(dim)
        return np.array(dims_of_labels)
    
    def dims_of_all_spaces(self):
        """
        
        Parameters
        ----------
        self
           
        Requires 
        --------
        nothing
    
        Returns
        -------
        dictionary of with the spaces and their respective dimensions
            
        """
        #get label and dimension information from comb
        (Names, dims) = (np.array(self.spaces), np.array(self.dims))
        #Create dictionary and print information
        Dict = {}
        for n in np.arange(len(Names)):
            (nam,d) = (Names[n],dims[n])
            print(str(nam) + ':', d)
            Dict[nam] = d
        return Dict
        
        
M1 = np.random.rand(144).reshape((12,12))
M2 = np.random.rand(24**2).reshape((24,24))
C1 = Comb(M1,[2,3,2],['A','B','C'])
C2 = Comb(M2,[2,3,2],['A','R','D'])



def wocb():
    """
    
    Parameters
    ----------
    none
    
    Requires
    -------
    numpy as np
    tn_product

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
    return 1./4*(tn_product(ident,ident,ident,ident) + 1/np.sqrt(2)*(tn_product(ident,sigmaZ,sigmaZ,ident) + tn_product(sigmaZ,ident,sigmaX,sigmaZ)))



def choi_to_vec(C,dims):
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
    
    
    
