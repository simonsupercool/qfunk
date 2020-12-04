#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020

@author: simon
"""
import numpy as np

def TrX(rho,sys,dim):
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
