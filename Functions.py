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
	#Function that computes the partial transpose of rho with respect to the 
	#systems given in sys. sys needs to be a list (starting with 0 for the 
	#first subsystem, even if there is only one system with respect to which 
	#the transpose is taken. dim is the array containing the dimensions of 
	#the subsystems.
	#
	#Requires: numpy
	

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
    #Function that computes the partial trace of rho with respect to the 
    #systems given in sys. sys needs to be a list (starting with 0 for the 
    #first subsystem), even if there is only one system with respect to which 
    #the transpose is taken. dim is the array containing the dimensions of 
    #the subsystems.
    #
    #Requires:  numpy
    
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

