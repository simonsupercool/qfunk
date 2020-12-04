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
