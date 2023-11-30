#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:57:21 2020
@author: simon
Contains basic functions for linear algebra. 
"""

import numpy as np
from scipy import sparse
from multiprocessing import Pool

import qfunk.opensys




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


def unitary_choi(U):
    """
    Parameters
    ----------
    U: dxd unitary matrix to compute choi form of
    
    Requires
    -------
    numpy as np
    qfunk.opensys as qop
    
    Returns
    -------
    A numpy matrix of d^2 x d^2 representing the Choi matrix (not state) of U. Acts on second half of 
    tensor product space. 
    
    """

    # get dimension 
    dim = len(U)

    # compute A-form of map
    A_form = np.kron(np.conj(U), U)

    # permute subsystem 
    B_form = qops.choi_involution(A_form, [dim,dim])

    return B_form
    

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




def tn_product(*args, sparse=False, list_of_mat=False):
    """
    Parameters
    ----------
    args: elements which are to be tensored
    
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
        if list_of_mat:         #account for tuples of matrices
            result = args[0][0]
            for Mat in args[0][1:]:
            if sparse:
                result = sparse.kron(result, Mat)
            else:
                result = np.kron(result,Mat)
        else: 
            result = args[0]
            for Mat in args[1:]:
                if sparse:
                    result = sparse.kron(result, Mat)
                else:
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
            set_el_sym = _Ejk(d, j, k) + _Ejk(d, k, j)

            # create antisymmetric component
            set_el_asym = -1j*(_Ejk(d, j, k) - _Ejk(d, k, j)) 

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


def oppressor(M,tol=1e-15):
    """
    Sets small values of array to zero

    Parameters
    -----------
    M:  Matrix to be suppressed
    tol: tolerance below which elements are zeroed
    
    Requires
    -----------
    numpy as np
    
    Returns
    -----------
    Suppressed array of same size as input
    """

    M[abs(M) < tol] = 0.0 

    return M

def mkron(A, n):
    """
    Computes the nth kronecker power of A^{\otimes n}
    
    Parameters
    -----------
    M:  Matrix to be raised to tensor power
    n: power of tensor product
    
    Requires
    -----------
    numpy as np
    
    Returns
    -----------
    Matrix whose dimensions are the same as the input but raised to the power n

    """

    # assign new variable
    B = A
    # iterate through powers
    for i in range(n-1):
        B = np.kron(B,A)
        
    return B



def _data_yielder(state, N, M):
    """
    Iterable function that stores shadow tomography data for multiprocessing purposes
    
    Parameters
    -----------
    state: Target state (pure state vector or density operator) to perform shadow tomography on  
    N: Integer number of qubits in system
    M: Integer number of clifford measurements to use for shadow tomography, controls the error and probability of failure
    
    Requires
    -----------
    numpy as np
    
    Returns
    -----------
    tuple of (state, N)

    """

    # store internal copies
    internal_state = np.asarray(state)
    qubit_num = N
    shadow_num = M

    # iterate until full tomography is complete
    n = 0
    while n <= shadow_num:
        yield (internal_state, qubit_num)
        n +=1





def _shadow_get(a):
    """
    Performs a single random Clifford measurement and returns it as a stabiliser tableau
    
    Parameters
    -----------
    a: Tuple of a state and the number of qubits it contains
    
    Requires
    -----------
    numpy as np
    
    Returns
    -----------
    A single Clifford measurement outcome of the state in stabliser form

    """

    state, sys_num = a

    #------------------------
    # sparse X gate
    X = csr_matrix(np.asarray([[0,1],[1,0]]))
    #------------------------

    # check if state vector or density operator
    if np.shape(state)[0] != np.shape(state)[1]:

        # generate clifford
        cliff_circ = random_clifford(sys_num)
        cliff_circ.phase = [False]*2*sys_num
        cliff_op = csr_matrix(Operator(cliff_circ).data)
        # apply to input vector
        cliff_vec = cliff_op @ state

        #-------------------------------------------------------
        
        # get the output probability distribution
        outcome_distribution = np.squeeze(abs(cliff_vec)**2)

        # produce a single outcome
        outcome = np.random.choice(a=2**sys_num, p=outcome_distribution)

        # convert outcome to binary in REVERSE order
        binary_string = f"{outcome:0{sys_num}b}"[::-1]

        #-------------------------------------------------------

        # compute and apply the operators to build correct binary string
        ops = [int(p) for p in binary_string]

        # initialise quantum circuit
        qc = QuantumCircuit(sys_num)
        for i,el in enumerate(ops):
          if el==1:
              qc.x(i)

        # convert to Clifford circuit
        cliff = qc
        # compute total clifford action
        clifford_circ = cliff.compose(cliff_circ.adjoint().to_circuit())

    # input is density operator
    else:

        # generate clifford
        cliff_circ = random_clifford(sys_num)
        cliff_circ.phase = [False]*2*sys_num
        cliff_op = csr_matrix(Operator(cliff_circ).data)
        # apply to input vector
        cliff_state = cliff_op @ state @ qut.dagger(cliff_op)

        #-------------------------------------------------------
        
        # get the output probability distribution
        outcome_distribution = np.squeeze(abs(np.diag(cliff_state)))

        # produce a single outcome
        outcome = np.random.choice(a=2**sys_num, p=outcome_distribution)

        # convert outcome to binary string in REVERSE order
        binary_string = f"{outcome:0{sys_num}b}"[::-1]

        #-------------------------------------------------------

        # compute and apply the operators to build correct binary string
        ops = [int(p) for p in binary_string]

        # initialise quantum circuit
        qc = QuantumCircuit(sys_num)
        for i,el in enumerate(ops):
          if el==1:
              qc.x(i)

        # convert to Clifford circuit
        cliff = qc
        # compute total clifford action
        clifford_circ = cliff.compose(cliff_circ.adjoint().to_circuit())

    # add the clifford operation and it's outcome to the shadow pair for reconstruction later
    return clifford_circ



def shadow_rebuild(state, sys_num, M, workers=1, silence=False):
    """
    Performs a shadow tomography on the target state consisting of <sys_num> qubits using <M> clifford measurements
    
    Parameters
    -----------
    state: Target state (pure state vector or density operator) to perform shadow tomography on  
    sys_num: Integer number of qubits in system
    M: Integer number of clifford measurements to use for shadow tomography, controls the error and probability of failure
    workers: Integer number of simulateous simulators to run to acquire shadows - approximately linear scaling speedup for increased memory usage
    silence: Boolean flag on whether to display progress
    
    Requires
    -----------
    numpy as np
    qiskit.quantum info
    
    Returns
    -----------
    List of operators U^dagger such that U^dagger|0> is the stabiliser measurement outcome 

    """

    # import optional qiskit library function for random generation
    from qiskit.quantum_info import random_clifford, Operator, Clifford

    #-------------------------------------------------------
    # generate the random Clifford measurements we will use
    shadows = []
    data_streamer = _data_yielder(state, sys_num, M)

    # compute shadows in parallel
    with Pool(processes=workers) as pool:
        shadows += list(tqdm(pool.imap_unordered(func=_shadow_get, iterable=data_streamer, chunksize=1), total=M, desc='Performing reconstruction from the shadow realm', ascii=' #', leave=False, disable=silence))
        
    return shadows



def shadow_estimator(shadows, sys_num):
    """
    Performs a shadow tomography on the target state consisting of <sys_num> qubits using <M> clifford measurements
    
    Parameters
    -----------
    state: Target state (pure state vector or density operator) to perform shadow tomography on  
    sys_num: Integer number of qubits in system
    
    Requires
    -----------
    numpy as np
    qiskit.quantum_info
    
    Returns
    -----------
    The classical shadow estimator
    """

    # import optional qiskit library function for random generation
    from qiskit.quantum_info import Operator

    # initialise density operator
    rho = np.zeros([2**sys_num]*2, dtype=np.complex128)
    # seed state for stabiliser states
    seed_state = np.zeros([2**sys_num]*2, dtype=np.complex128)
    seed_state[0,0] = 1.0

    # iterate over all measured shadows
    for shad in tqdm(shadows, desc='Generating mixed shadow state'):
        U = Operator(shad).data
        rho += U @ seed_state @ qut.dagger(U)

    # compute the average state
    rho /= len(shadows)

    # return the estimator
    return (2**sys_num + 1)*rho - np.eye(2**sys_num)



